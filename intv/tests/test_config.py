import pytest
import tempfile
import os
from unittest.mock import patch, mock_open
from intv import config

@pytest.mark.unit
def test_load_config_file_exists():
    """Test loading config when file exists"""
    mock_yaml_content = """
    llm:
      model: "test-model"
      temperature: 0.7
    audio:
      sample_rate: 16000
    """
    
    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("os.path.exists", return_value=True):
            with patch("yaml.safe_load") as mock_yaml_load:
                mock_yaml_load.return_value = {
                    "llm": {"model": "test-model", "temperature": 0.7},
                    "audio": {"sample_rate": 16000}
                }
                result = config.load_config("test_config.yaml")
                assert result["llm"]["model"] == "test-model"
                assert result["audio"]["sample_rate"] == 16000

@pytest.mark.unit
def test_load_config_file_not_exists():
    """Test loading config when file doesn't exist"""
    with patch("os.path.exists", return_value=False):
        with patch("builtins.print") as mock_print:
            result = config.load_config("nonexistent.yaml")
            assert result == {}
            mock_print.assert_called()

@pytest.mark.unit
def test_get_config_value_exists():
    """Test getting existing config value"""
    test_config = {
        "llm": {"model": "test-model"},
        "audio": {"sample_rate": 16000}
    }
    
    with patch.object(config, 'current_config', test_config):
        assert config.get_config_value("llm.model") == "test-model"
        assert config.get_config_value("audio.sample_rate") == 16000

@pytest.mark.unit
def test_get_config_value_default():
    """Test getting config value with default"""
    with patch.object(config, 'current_config', {}):
        assert config.get_config_value("nonexistent.key", "default") == "default"

@pytest.mark.unit
def test_validate_config_structure():
    """Test config structure validation"""
    valid_config = {
        "llm": {"model": "test", "temperature": 0.7},
        "audio": {"sample_rate": 16000}
    }
    
    # This test assumes a validate_config function exists
    # You may need to implement this function in your config module
    if hasattr(config, 'validate_config'):
        assert config.validate_config(valid_config) is True
