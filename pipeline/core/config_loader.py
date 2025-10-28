import yaml
import logging
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded config from: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'pipeline': {
                'max_execution_time': 3600,
                'log_level': 'INFO'
            },
            'cache': {
                'directory': 'cache',
                'max_size_mb': 100
            },
            'epochs': {
                'directory': 'epochs',
                'max_epochs': 50
            },
            'routing': {
                'default_speed_urban': 60,
                'default_speed_regional': 100,
                'default_speed_high': 200
            }
        }