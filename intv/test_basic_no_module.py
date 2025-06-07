#!/usr/bin/env python3
"""
Test basic pipeline functionality without modules
"""

import os
import tempfile
import subprocess
import sys

def test_basic_pipeline_no_module():
    """Test that the pipeline works with basic document processing without modules"""
    print("Testing basic pipeline without module...")
    
    # Create a test document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for basic processing.\n")
        f.write("It contains some simple text that should be processed.\n")
        f.write("The pipeline should work without requiring any modules.\n")
        test_file = f.name
    
    try:
        # Test basic processing without --module argument
        cmd = [
            sys.executable, "-m", "intv.pipeline_cli",
            "--files", test_file,
            "--output", "test_output.txt"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check if command succeeded
        if result.returncode == 0:
            print("✅ Basic pipeline without module works!")
            return True
        else:
            print("❌ Basic pipeline without module failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_help_output():
    """Test that help output shows module as optional"""
    print("\nTesting help output...")
    
    cmd = [sys.executable, "-m", "intv.pipeline_cli", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("Help output:")
    print(result.stdout)
    
    # Check if help indicates module is optional
    if "optional" in result.stdout.lower() and "module" in result.stdout.lower():
        print("✅ Help shows module as optional")
        return True
    else:
        print("❌ Help doesn't clearly show module as optional")
        return False

if __name__ == "__main__":
    print("Testing basic pipeline functionality without modules\n")
    
    success1 = test_help_output()
    success2 = test_basic_pipeline_no_module()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
