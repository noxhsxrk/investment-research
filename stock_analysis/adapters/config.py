"""Configuration management for data source adapters.

This module provides configuration management for different data source
adapters, including credential management and adapter-specific settings.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml
import json

from stock_analysis.utils.config import config as app_config
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for a data source adapter."""
    
    name: str
    enabled: bool = True
    priority: int = 1  # Lower numbers = higher priority
    rate_limit: float = 1.0  # Requests per second
    timeout: int = 30
    max_retries: int = 3
    credentials: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def get_credential(self, key: str) -> Optional[str]:
        """Get credential value, checking environment variables first.
        
        Args:
            key: Credential key name
            
        Returns:
            Credential value or None if not found
        """
        # Check environment variable first (with adapter name prefix)
        env_key = f"{self.name.upper()}_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
        
        # Check generic environment variable
        env_value = os.getenv(key.upper())
        if env_value:
            return env_value
        
        # Check stored credentials
        return self.credentials.get(key)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get adapter setting value.
        
        Args:
            key: Setting key name
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)


class AdapterConfigManager:
    """Manager for adapter configurations."""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or self._get_default_config_file()
        self.configs: Dict[str, AdapterConfig] = {}
        self._load_configs()
    
    def _get_default_config_file(self) -> str:
        """Get default configuration file path."""
        # Try to get from app config first
        config_file = app_config.get('stock_analysis.adapters.config_file')
        if config_file:
            return config_file
        
        # Default to adapters.yaml in the same directory as the main config
        return 'adapters.yaml'
    
    def _load_configs(self) -> None:
        """Load adapter configurations from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    if self.config_file.endswith('.json'):
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)
                
                self._parse_config_data(data)
                logger.info(f"Loaded adapter configurations from {self.config_file}")
            else:
                logger.info(f"Configuration file {self.config_file} not found, using defaults")
                self._load_default_configs()
        except Exception as e:
            logger.error(f"Error loading adapter configurations: {str(e)}")
            self._load_default_configs()
    
    def _parse_config_data(self, data: Dict[str, Any]) -> None:
        """Parse configuration data and create AdapterConfig objects."""
        adapters_data = data.get('adapters', {})
        
        for adapter_name, adapter_data in adapters_data.items():
            config = AdapterConfig(
                name=adapter_name,
                enabled=adapter_data.get('enabled', True),
                priority=adapter_data.get('priority', 1),
                rate_limit=adapter_data.get('rate_limit', 1.0),
                timeout=adapter_data.get('timeout', 30),
                max_retries=adapter_data.get('max_retries', 3),
                credentials=adapter_data.get('credentials', {}),
                settings=adapter_data.get('settings', {})
            )
            self.configs[adapter_name] = config
    
    def _load_default_configs(self) -> None:
        """Load default configurations for known adapters."""
        default_configs = {
            'yfinance': AdapterConfig(
                name='yfinance',
                enabled=True,
                priority=1,
                rate_limit=1.0,
                timeout=30,
                max_retries=3,
                settings={
                    'cache_enabled': True,
                    'cache_ttl': 300  # 5 minutes
                }
            ),
            'investing': AdapterConfig(
                name='investing',
                enabled=False,  # Disabled by default until implemented
                priority=2,
                rate_limit=0.5,  # More conservative rate limiting
                timeout=45,
                max_retries=3,
                settings={
                    'cache_enabled': True,
                    'cache_ttl': 600  # 10 minutes
                }
            ),
            'alpha_vantage': AdapterConfig(
                name='alpha_vantage',
                enabled=False,  # Disabled by default until implemented
                priority=3,
                rate_limit=0.2,  # Very conservative for free tier
                timeout=30,
                max_retries=3,
                credentials={
                    'api_key': ''  # Must be provided via environment or config
                },
                settings={
                    'cache_enabled': True,
                    'cache_ttl': 900  # 15 minutes
                }
            )
        }
        
        self.configs.update(default_configs)
        logger.info("Loaded default adapter configurations")
    
    def get_config(self, adapter_name: str) -> Optional[AdapterConfig]:
        """Get configuration for a specific adapter.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            AdapterConfig object or None if not found
        """
        return self.configs.get(adapter_name)
    
    def get_enabled_adapters(self) -> Dict[str, AdapterConfig]:
        """Get all enabled adapter configurations.
        
        Returns:
            Dictionary of enabled adapter configurations
        """
        return {name: config for name, config in self.configs.items() if config.enabled}
    
    def get_adapters_by_priority(self) -> List[AdapterConfig]:
        """Get enabled adapters sorted by priority.
        
        Returns:
            List of adapter configurations sorted by priority (lower number = higher priority)
        """
        enabled_configs = list(self.get_enabled_adapters().values())
        return sorted(enabled_configs, key=lambda x: x.priority)
    
    def add_config(self, config: AdapterConfig) -> None:
        """Add or update adapter configuration.
        
        Args:
            config: AdapterConfig object to add
        """
        self.configs[config.name] = config
        logger.info(f"Added/updated configuration for adapter: {config.name}")
    
    def save_configs(self) -> None:
        """Save current configurations to file."""
        try:
            config_data = {
                'adapters': {}
            }
            
            for name, config in self.configs.items():
                config_data['adapters'][name] = {
                    'enabled': config.enabled,
                    'priority': config.priority,
                    'rate_limit': config.rate_limit,
                    'timeout': config.timeout,
                    'max_retries': config.max_retries,
                    'credentials': config.credentials,
                    'settings': config.settings
                }
            
            with open(self.config_file, 'w') as f:
                if self.config_file.endswith('.json'):
                    json.dump(config_data, f, indent=2)
                else:
                    yaml.dump(config_data, f, default_flow_style=False)
            
            logger.info(f"Saved adapter configurations to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving adapter configurations: {str(e)}")
    
    def validate_configs(self) -> Dict[str, List[str]]:
        """Validate all adapter configurations.
        
        Returns:
            Dictionary with validation results for each adapter
        """
        validation_results = {}
        
        for name, config in self.configs.items():
            issues = []
            
            # Check required credentials for specific adapters
            if name == 'alpha_vantage' and config.enabled:
                if not config.get_credential('api_key'):
                    issues.append("Missing required API key")
            
            if name == 'investing' and config.enabled:
                # Add specific validation for investing.com adapter when implemented
                pass
            
            # Check rate limiting settings
            if config.rate_limit <= 0:
                issues.append("Rate limit must be positive")
            
            if config.timeout <= 0:
                issues.append("Timeout must be positive")
            
            if config.max_retries < 0:
                issues.append("Max retries cannot be negative")
            
            validation_results[name] = issues
        
        return validation_results


# Global configuration manager instance
_config_manager = None


def get_adapter_config_manager() -> AdapterConfigManager:
    """Get the global adapter configuration manager instance.
    
    Returns:
        AdapterConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = AdapterConfigManager()
    return _config_manager


def get_adapter_config(adapter_name: str) -> Optional[AdapterConfig]:
    """Get configuration for a specific adapter.
    
    Args:
        adapter_name: Name of the adapter
        
    Returns:
        AdapterConfig object or None if not found
    """
    manager = get_adapter_config_manager()
    return manager.get_config(adapter_name)