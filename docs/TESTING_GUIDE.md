# Testing Guide for Interview Application

## Overview

This document provides comprehensive testing guidance for the Interview Application project. All 4 testing recommendations have been successfully implemented.

## ✅ Completed Implementations

### 1. **Existing Tests Status** ✅
- **Fixed test failures** in `test_server_utils.py`
- **Enhanced test coverage** with proper mocking and edge cases
- **Added proper socket handling** for port testing

### 2. **Testing Configuration** ✅
- **Created `pytest.ini`** with comprehensive configuration
- **Added custom markers** for test categorization (unit, integration, slow, network, audio, llm, cli)
- **Configured coverage reporting** with HTML and terminal output
- **Set up test timeouts** and warning filters

### 3. **Enhanced Test Coverage** ✅
- **Expanded `test_server_utils.py`** with 12 comprehensive tests
- **Added proper mocking** using `unittest.mock`
- **Categorized tests** with appropriate markers
- **Added edge case testing** and error handling tests

### 4. **Additional Test Files** ✅
Created comprehensive test suites for key modules:
- **`test_config.py`** - Configuration management tests
- **`test_audio_utils.py`** - Audio processing functionality tests
- **`test_utils.py`** - General utility function tests  
- **`test_llm.py`** - Language model integration tests
- **`test_cli.py`** - Command line interface tests

## 📁 Test File Structure

```
tests/
├── __init__.py             # Tests package initialization
├── test_server_utils.py    # Server utilities (enhanced)
├── test_config.py          # Configuration management
├── test_audio_utils.py     # Audio processing
├── test_utils.py           # General utilities
├── test_llm.py            # LLM integration
└── test_cli.py            # CLI interface
```

## 🚀 Running Tests

### Quick Start
```bash
# Run all tests
cd /home/nyx/INTV_Apps/INTV
./run_tests.sh

# Run specific test categories
./run_tests.sh unit         # Unit tests only
./run_tests.sh integration  # Integration tests only
./run_tests.sh fast         # Fast tests (exclude slow ones)
./run_tests.sh coverage     # Tests with coverage report
```

### Manual pytest Commands
```bash
# Activate virtual environment and set PYTHONPATH
cd /home/nyx/INTV_Apps/INTV
source .venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run all tests
pytest tests/test_*.py -v

# Run specific test file
pytest tests/test_server_utils.py -v

# Run tests by category
pytest -m unit -v                 # Unit tests only
pytest -m "not slow" -v          # Fast tests only
pytest -m "audio or llm" -v      # Audio and LLM tests

# Run with coverage
pytest --cov=intv --cov-report=html tests/test_*.py
```

## 📊 Test Categories

### 🔹 Unit Tests (`@pytest.mark.unit`)
- **Fast, isolated tests**
- Test individual functions/methods
- Use mocking for external dependencies
- Examples: config loading, utility functions, port checking

### 🔹 Integration Tests (`@pytest.mark.integration`)  
- **Test component interactions**
- May use external services (mocked)
- Examples: server startup/shutdown, service orchestration

### 🔹 Slow Tests (`@pytest.mark.slow`)
- **Long-running tests**
- File operations, model loading
- Excluded in fast test runs

### 🔹 Network Tests (`@pytest.mark.network`)
- **Require network connectivity**
- API calls, health checks
- Can be skipped in offline environments

### 🔹 Audio Tests (`@pytest.mark.audio`)
- **Audio processing functionality**
- Audio loading, processing, feature extraction

### 🔹 LLM Tests (`@pytest.mark.llm`) 
- **Language model operations**
- Model loading, text generation, encoding/decoding

### 🔹 CLI Tests (`@pytest.mark.cli`)
- **Command line interface**
- Argument parsing, command execution

## 📈 Coverage Reporting

### HTML Coverage Report
```bash
./run_tests.sh coverage
# Opens: htmlcov/index.html
```

### Terminal Coverage
```bash
pytest --cov=intv --cov-report=term-missing intv/test_*.py
```

## 🛠️ Test Development Guidelines

### Writing New Tests
1. **Choose appropriate markers** (`@pytest.mark.unit`, etc.)
2. **Use descriptive test names** (`test_function_scenario_expected`)
3. **Mock external dependencies** properly
4. **Test both success and failure cases**
5. **Include edge cases and boundary conditions**

### Example Test Structure
```python
@pytest.mark.unit
def test_function_name_scenario():
    """Clear description of what is being tested"""
    # Arrange
    test_data = setup_test_data()
    
    # Act  
    with patch('module.external_dependency') as mock_dep:
        result = function_under_test(test_data)
    
    # Assert
    assert result == expected_value
    mock_dep.assert_called_once()
```

### Mocking Best Practices
```python
# Mock external dependencies
with patch('requests.get') as mock_get:
    mock_get.return_value.status_code = 200
    
# Mock internal modules
with patch.object(module, 'function', return_value='mocked'):
    
# Mock file operations  
with patch('builtins.open', mock_open(read_data='test')):
```

## 🔧 Configuration Files

### `pytest.ini`
- Test discovery settings
- Output formatting
- Custom markers registration  
- Coverage configuration
- Warning filters

### `run_tests.sh`
- Convenient test runner script
- Category-based test execution
- Colored output and progress reporting
- Virtual environment handling

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Ensure `PYTHONPATH` includes `src` directory
2. **Module not found**: Activate virtual environment first
3. **Test timeouts**: Check network connectivity for network tests
4. **Coverage not working**: Install `pytest-cov` plugin

### Debugging Tests
```bash
# Run with verbose output and no capture
pytest -v -s tests/test_file.py::test_function

# Run with debugging breakpoints
pytest --pdb tests/test_file.py::test_function

# Run and stop on first failure
pytest -x tests/test_*.py
```

## 📋 Next Steps

### Recommended Additions
1. **End-to-end tests** for complete workflows
2. **Performance tests** for audio/LLM operations  
3. **API tests** for FastAPI endpoints
4. **Database tests** if data persistence is added
5. **Docker container tests** for deployment validation

### Continuous Integration
Consider setting up CI/CD pipeline with:
- Automated test execution on code changes
- Coverage reporting and badges
- Test result notifications
- Parallel test execution for faster feedback

## 📚 Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-cov Coverage](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**Test Suite Status**: ✅ All 4 recommendations implemented and operational
