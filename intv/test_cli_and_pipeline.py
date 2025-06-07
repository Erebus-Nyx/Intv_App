import subprocess
import sys
import os
import pytest

def test_cli_help():
    result = subprocess.run([sys.executable, 'src/main.py', '--help'], capture_output=True, text=True)
    assert 'usage' in result.stdout.lower()
    assert result.returncode == 0

def test_cli_version():
    result = subprocess.run([sys.executable, 'src/main.py', '--version'], capture_output=True, text=True)
    assert 'intv cli version' in result.stdout.lower()
    assert result.returncode == 0

def test_cli_no_args():
    result = subprocess.run([sys.executable, 'src/main.py'], capture_output=True, text=True)
    assert 'select a file to analyze' in result.stdout.lower() or 'no file selected' in result.stdout.lower() or 'select an interview/module type' in result.stdout.lower()

# Add more tests for error handling, e.g. invalid file, multiple audio sources, etc.
def test_cli_invalid_file():
    result = subprocess.run([sys.executable, 'src/main.py', '--file', 'not_a_real_file.pdf'], capture_output=True, text=True)
    assert 'invalid file' in result.stdout.lower() or result.returncode != 0
