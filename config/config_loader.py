import yaml
from pathlib import Path
from typing import Dict
import logging


class ConfigLoader:
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or str(Path(__file__).parent / "config.yaml")

    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def save_config(self, config: Dict, path: str = None):
        """Save configuration to YAML file"""
        save_path = path or self.config_path
        try:
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
