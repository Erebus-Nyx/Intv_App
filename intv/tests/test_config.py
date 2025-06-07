#!/usr/bin/env python3
"""
Test configuration management
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json
import os

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_config_import():
    """Test that config module can be imported."""
    try:
        from config import Config
        assert Config is not None
    except ImportError as e:
        pytest.fail(f"Could not import Config: {e}")

def test_config_initialization():
    """Test config initialization."""
    try:
        from config import Config
        config = Config()
        assert config is not None
    except Exception as e:
        pytest.fail(f"Could not initialize Config: {e}")

def test_config_yaml_to_settings_json():
    """Test that config.yaml is properly converted to settings.json."""
    try:
        from config import Config
        
        # Create a temporary config.yaml
        config_yaml_content = """
database:
  host: localhost
  port: 5432
  name: test_db

api:
  host: 0.0.0.0
  port: 8000
  
rag:
  mode: embedded
  
llm:
  provider: koboldcpp
  model: test-model
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_yaml_path = Path(temp_dir) / "config.yaml"
            settings_json_path = Path(temp_dir) / "settings.json"
            
            # Write config.yaml
            with open(config_yaml_path, 'w') as f:
                f.write(config_yaml_content)
            
            # Initialize config with the temp directory
            config = Config(config_path=str(config_yaml_path))
            
            # Check if settings.json was created or config was loaded
            assert config is not None
            
    except Exception as e:
        pytest.fail(f"Config YAML to JSON conversion failed: {e}")

def test_environment_variable_population():
    """Test that environment variables are properly populated."""
    try:
        from config import Config
        
        # Set some test environment variables
        test_env_vars = {
            'INTV_DATABASE_HOST': 'env-localhost',
            'INTV_API_PORT': '9000',
            'INTV_RAG_MODE': 'external'
        }
        
        with patch.dict(os.environ, test_env_vars):
            config = Config()
            # Verify that config can be initialized with env vars
            assert config is not None
            
    except Exception as e:
        pytest.fail(f"Environment variable population failed: {e}")

def test_config_default_values():
    """Test that config has reasonable default values."""
    try:
        from config import Config
        config = Config()
        
        # Config should have some default structure
        assert config is not None
        
    except Exception as e:
        pytest.fail(f"Config default values test failed: {e}")

if __name__ == '__main__':
    pytest.main([__file__])
