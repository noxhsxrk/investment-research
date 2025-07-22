"""Configuration management for stock analysis system."""

import os
import yaml
import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from stock_analysis.utils.exceptions import ConfigurationError


class ConfigManager:
    """Manages application configuration from YAML files and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. Defaults to 'config.yaml'
        """
        self.config_path = config_path or "config.yaml"
        self._config = {}
        self._schema = {}
        self._last_modified_time = 0
        self._watch_thread = None
        self._watch_interval = 5  # seconds
        self._watching = False
        self._callbacks: List[Callable[[], None]] = []
        self._load_schema()
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Load default configuration
        self._config = self._get_default_config()
        
        # Override with file configuration if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as file:
                    file_config = yaml.safe_load(file) or {}
                    self._merge_config(self._config, file_config)
                self._last_modified_time = os.path.getmtime(self.config_path)
            except Exception as e:
                raise ConfigurationError(f"Failed to load configuration from {self.config_path}: {str(e)}")
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Validate configuration
        self._validate_config()
    
    def _load_schema(self) -> None:
        """Load configuration schema for validation."""
        self._schema = {
            'stock_analysis': {
                'data_sources': {
                    'yfinance': {
                        'timeout': {'type': 'int', 'min': 1, 'max': 300},
                        'retry_attempts': {'type': 'int', 'min': 0, 'max': 10},
                        'rate_limit_delay': {'type': 'float', 'min': 0.0, 'max': 10.0}
                    }
                },
                'analysis': {
                    'default_period': {'type': 'str', 'allowed': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']},
                    'valuation_models': {'type': 'list', 'allowed_items': ['dcf', 'peer_comparison', 'peg', 'dividend_discount']},
                    'sentiment_sources': {'type': 'list', 'allowed_items': ['yahoo_news', 'google_news', 'twitter', 'reddit']}
                },
                'export': {
                    'default_format': {'type': 'str', 'allowed': ['csv', 'excel', 'json']},
                    'output_directory': {'type': 'str'},
                    'include_charts': {'type': 'bool'}
                },
                'scheduling': {
                    'default_interval': {'type': 'str', 'allowed': ['hourly', 'daily', 'weekly', 'monthly']},
                    'notification_email': {'type': 'str', 'nullable': True}
                },
                'logging': {
                    'level': {'type': 'str', 'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']},
                    'file_path': {'type': 'str'},
                    'max_file_size': {'type': 'str'},
                    'backup_count': {'type': 'int', 'min': 0, 'max': 100}
                },
                'stock_lists': {
                    'default_files': {'type': 'list'},
                    'auto_validate': {'type': 'bool', 'default': False},
                    'cache_industry_data': {'type': 'bool', 'default': True}
                },
                'cache': {
                    'use_disk_cache': {'type': 'bool', 'default': True},
                    'directory': {'type': 'str', 'default': './.cache'},
                    'max_memory_size': {'type': 'int', 'min': 1024 * 1024, 'max': 1024 * 1024 * 1024, 'default': 100 * 1024 * 1024},
                    'max_disk_size': {'type': 'int', 'min': 1024 * 1024, 'max': 1024 * 1024 * 1024 * 10, 'default': 1024 * 1024 * 1024},
                    'cleanup_interval': {'type': 'int', 'min': 60, 'max': 86400, 'default': 3600},
                    'expiry': {
                        'stock_info': {'type': 'int', 'min': 60, 'max': 86400 * 30, 'default': 3600},
                        'historical_data': {'type': 'int', 'min': 60, 'max': 86400 * 30, 'default': 86400},
                        'financial_statements': {'type': 'int', 'min': 60, 'max': 86400 * 30, 'default': 86400 * 7},
                        'news': {'type': 'int', 'min': 60, 'max': 86400 * 30, 'default': 1800},
                        'peer_data': {'type': 'int', 'min': 60, 'max': 86400 * 30, 'default': 86400 * 7},
                        'general': {'type': 'int', 'min': 60, 'max': 86400 * 30, 'default': 3600}
                    }
                }
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'stock_analysis.data_sources.yfinance.timeout')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-notation key.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Args:
            path: Path to save configuration. Defaults to current config_path
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'stock_analysis': {
                'data_sources': {
                    'yfinance': {
                        'timeout': 30,
                        'retry_attempts': 3,
                        'rate_limit_delay': 1.0
                    }
                },
                'analysis': {
                    'default_period': '1y',
                    'valuation_models': ['dcf', 'peer_comparison', 'peg'],
                    'sentiment_sources': ['yahoo_news', 'google_news']
                },
                'export': {
                    'default_format': 'excel',
                    'output_directory': './exports',
                    'include_charts': True
                },
                'scheduling': {
                    'default_interval': 'daily',
                    'notification_email': None
                },
                'logging': {
                    'level': 'INFO',
                    'file_path': './logs/stock_analysis.log',
                    'max_file_size': '10MB',
                    'backup_count': 5
                },
                'cache': {
                    'use_disk_cache': True,
                    'directory': './.cache',
                    'max_memory_size': 100 * 1024 * 1024,  # 100MB
                    'max_disk_size': 1024 * 1024 * 1024,  # 1GB
                    'cleanup_interval': 3600,  # 1 hour
                    'expiry': {
                        'stock_info': 3600,  # 1 hour
                        'historical_data': 86400,  # 24 hours
                        'financial_statements': 86400 * 7,  # 1 week
                        'news': 1800,  # 30 minutes
                        'peer_data': 86400 * 7,  # 1 week
                        'general': 3600  # 1 hour
                    }
                }
            }
        }
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'STOCK_ANALYSIS_YFINANCE_TIMEOUT': 'stock_analysis.data_sources.yfinance.timeout',
            'STOCK_ANALYSIS_LOG_LEVEL': 'stock_analysis.logging.level',
            'STOCK_ANALYSIS_OUTPUT_DIR': 'stock_analysis.export.output_directory',
            'STOCK_ANALYSIS_NOTIFICATION_EMAIL': 'stock_analysis.scheduling.notification_email'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)
                
                self.set(config_key, value)
    
    def _is_float(self, value: str) -> bool:
        """Check if string can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _validate_config(self) -> None:
        """Validate configuration against schema."""
        errors = []
        self._validate_against_schema(self._config, self._schema, '', errors)
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            raise ConfigurationError(error_msg)
    
    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any], 
                                path: str, errors: List[str]) -> None:
        """Recursively validate configuration against schema."""
        for key, schema_value in schema.items():
            full_path = f"{path}.{key}" if path else key
            
            # Check if required key exists
            if key not in config:
                if isinstance(schema_value, dict) and schema_value.get('default') is not None:
                    # Set default value
                    config[key] = schema_value['default']
                elif isinstance(schema_value, dict) and schema_value.get('nullable', False):
                    # Nullable field can be missing
                    continue
                elif isinstance(schema_value, dict) and not any(k in schema_value for k in ['type', 'allowed', 'min', 'max']):
                    # It's a nested schema, create empty dict
                    config[key] = {}
                    self._validate_against_schema(config[key], schema_value, full_path, errors)
                continue
            
            value = config[key]
            
            # If it's a nested schema
            if isinstance(schema_value, dict) and not any(k in schema_value for k in ['type', 'allowed', 'min', 'max']):
                if not isinstance(value, dict):
                    errors.append(f"{full_path} should be an object")
                else:
                    self._validate_against_schema(value, schema_value, full_path, errors)
                continue
            
            # Validate type
            if 'type' in schema_value:
                if schema_value['type'] == 'str' and not isinstance(value, str):
                    if schema_value.get('nullable', False) and value is None:
                        continue
                    errors.append(f"{full_path} should be a string")
                elif schema_value['type'] == 'int' and not isinstance(value, int):
                    errors.append(f"{full_path} should be an integer")
                elif schema_value['type'] == 'float' and not isinstance(value, (int, float)):
                    errors.append(f"{full_path} should be a number")
                elif schema_value['type'] == 'bool' and not isinstance(value, bool):
                    errors.append(f"{full_path} should be a boolean")
                elif schema_value['type'] == 'list' and not isinstance(value, list):
                    errors.append(f"{full_path} should be a list")
            
            # Validate allowed values
            if 'allowed' in schema_value and value not in schema_value['allowed']:
                errors.append(f"{full_path} should be one of {schema_value['allowed']}")
            
            # Validate allowed items in list
            if 'allowed_items' in schema_value and isinstance(value, list):
                invalid_items = [item for item in value if item not in schema_value['allowed_items']]
                if invalid_items:
                    errors.append(f"{full_path} contains invalid items: {invalid_items}")
            
            # Validate min/max for numbers
            if isinstance(value, (int, float)):
                if 'min' in schema_value and value < schema_value['min']:
                    errors.append(f"{full_path} should be at least {schema_value['min']}")
                if 'max' in schema_value and value > schema_value['max']:
                    errors.append(f"{full_path} should be at most {schema_value['max']}")
    
    def start_watching(self) -> None:
        """Start watching configuration file for changes."""
        if self._watching:
            return
        
        self._watching = True
        self._watch_thread = threading.Thread(target=self._watch_config_file, daemon=True)
        self._watch_thread.start()
    
    def stop_watching(self) -> None:
        """Stop watching configuration file for changes."""
        self._watching = False
        if self._watch_thread:
            self._watch_thread.join(timeout=1.0)
            self._watch_thread = None
    
    def _watch_config_file(self) -> None:
        """Watch configuration file for changes and reload when modified."""
        while self._watching:
            try:
                if os.path.exists(self.config_path):
                    mod_time = os.path.getmtime(self.config_path)
                    if mod_time > self._last_modified_time:
                        self.load_config()
                        self._notify_callbacks()
                        self._last_modified_time = mod_time
            except Exception as e:
                # Log error but continue watching
                import logging
                logging.error(f"Error watching config file: {str(e)}")
            
            time.sleep(self._watch_interval)
    
    def register_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when configuration changes.
        
        Args:
            callback: Function to call when configuration changes
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[], None]) -> None:
        """Unregister a previously registered callback.
        
        Args:
            callback: Function to unregister
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks of configuration changes."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                import logging
                logging.error(f"Error in configuration change callback: {str(e)}")


# Global configuration instance
config = ConfigManager()