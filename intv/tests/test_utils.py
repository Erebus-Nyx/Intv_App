import pytest
import tempfile
import os
import json
from unittest.mock import patch, mock_open, Mock
from intv import utils

@pytest.mark.unit
def test_load_json_file_success():
    """Test successful JSON file loading"""
    test_data = {"key": "value", "number": 42}
    
    with patch("builtins.open", mock_open(read_data=json.dumps(test_data))):
        with patch("os.path.exists", return_value=True):
            result = utils.load_json_file("test.json")
            assert result == test_data

@pytest.mark.unit
def test_load_json_file_not_found():
    """Test JSON file loading when file doesn't exist"""
    with patch("os.path.exists", return_value=False):
        result = utils.load_json_file("nonexistent.json")
        assert result is None

@pytest.mark.unit
def test_load_json_file_invalid_json():
    """Test JSON file loading with invalid JSON"""
    with patch("builtins.open", mock_open(read_data="invalid json")):
        with patch("os.path.exists", return_value=True):
            with pytest.raises(json.JSONDecodeError):
                utils.load_json_file("invalid.json")

@pytest.mark.unit
def test_save_json_file():
    """Test saving data to JSON file"""
    test_data = {"test": "data", "numbers": [1, 2, 3]}
    
    with patch("builtins.open", mock_open()) as mock_file:
        utils.save_json_file(test_data, "output.json")
        mock_file.assert_called_once_with("output.json", "w", encoding="utf-8")
        # Check that json.dump was called with correct data
        handle = mock_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        assert "test" in written_data
        assert "data" in written_data

@pytest.mark.unit
def test_ensure_directory_exists():
    """Test directory creation"""
    test_dir = "/tmp/test_dir"
    
    with patch("os.makedirs") as mock_makedirs:
        with patch("os.path.exists", return_value=False):
            utils.ensure_directory_exists(test_dir)
            mock_makedirs.assert_called_once_with(test_dir, exist_ok=True)

@pytest.mark.unit
def test_ensure_directory_exists_already_exists():
    """Test directory creation when directory already exists"""
    test_dir = "/tmp/existing_dir"
    
    with patch("os.makedirs") as mock_makedirs:
        with patch("os.path.exists", return_value=True):
            utils.ensure_directory_exists(test_dir)
            # Should still call makedirs with exist_ok=True
            mock_makedirs.assert_called_once_with(test_dir, exist_ok=True)

@pytest.mark.unit
def test_format_timestamp():
    """Test timestamp formatting"""
    timestamp = 1625097600  # July 1, 2021 00:00:00 UTC
    
    formatted = utils.format_timestamp(timestamp)
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    # Basic check that it looks like a timestamp
    assert any(char.isdigit() for char in formatted)

@pytest.mark.unit
def test_sanitize_filename():
    """Test filename sanitization"""
    dangerous_name = "file<>:\"/\\|?*name.txt"
    sanitized = utils.sanitize_filename(dangerous_name)
    
    # Should not contain dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        assert char not in sanitized
    
    # Should still contain the basic filename parts
    assert "file" in sanitized
    assert "name" in sanitized
    assert ".txt" in sanitized

@pytest.mark.unit
def test_truncate_text():
    """Test text truncation"""
    long_text = "This is a very long text that should be truncated"
    max_length = 20
    
    truncated = utils.truncate_text(long_text, max_length)
    assert len(truncated) <= max_length
    assert truncated.endswith("...") if len(long_text) > max_length else True

@pytest.mark.unit
def test_truncate_text_short():
    """Test text truncation with short text"""
    short_text = "Short text"
    max_length = 20
    
    truncated = utils.truncate_text(short_text, max_length)
    assert truncated == short_text  # Should be unchanged

@pytest.mark.unit
def test_merge_dicts():
    """Test dictionary merging"""
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {"b": {"d": 3}, "e": 4}
    
    merged = utils.merge_dicts(dict1, dict2)
    
    assert merged["a"] == 1
    assert merged["e"] == 4
    assert merged["b"]["c"] == 2
    assert merged["b"]["d"] == 3

@pytest.mark.unit
def test_validate_email():
    """Test email validation"""
    valid_emails = [
        "test@example.com",
        "user.name@domain.co.uk",
        "user+tag@example.org"
    ]
    
    invalid_emails = [
        "invalid-email",
        "@domain.com",
        "user@",
        "user@domain",
        ""
    ]
    
    for email in valid_emails:
        assert utils.validate_email(email) is True
    
    for email in invalid_emails:
        assert utils.validate_email(email) is False

@pytest.mark.unit
def test_retry_on_exception():
    """Test retry decorator functionality"""
    call_count = 0
    
    @utils.retry_on_exception(max_retries=3, delay=0.1)
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "success"
    
    result = failing_function()
    assert result == "success"
    assert call_count == 3

@pytest.mark.unit
def test_retry_on_exception_max_retries():
    """Test retry decorator with max retries exceeded"""
    call_count = 0
    
    @utils.retry_on_exception(max_retries=2, delay=0.1)
    def always_failing_function():
        nonlocal call_count
        call_count += 1
        raise Exception("Always fails")
    
    with pytest.raises(Exception, match="Always fails"):
        always_failing_function()
    
    assert call_count == 3  # Initial call + 2 retries
