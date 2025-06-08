#!/usr/bin/env python3
"""
Debug script for dependency manager issues
"""

import sys
import traceback

print("=== Testing dependency_manager.py imports ===")

try:
    print("1. Testing direct file execution...")
    with open('intv/dependency_manager.py', 'r') as f:
        content = f.read()
    
    print(f"File length: {len(content)} characters")
    print(f"Lines: {len(content.splitlines())}")
    
    # Check for obvious syntax issues
    try:
        compile(content, 'intv/dependency_manager.py', 'exec')
        print("✓ File compiles successfully")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        print(f"Line {e.lineno}: {e.text}")
        sys.exit(1)
    
    print("\n2. Testing module import...")
    import intv.dependency_manager as dm
    
    print("✓ Module imported")
    print(f"Module file: {dm.__file__}")
    print(f"Available attributes: {[attr for attr in dir(dm) if not attr.startswith('_')]}")
    
    print("\n3. Testing direct execution...")
    exec(content)
    print("✓ Direct execution successful")
    
    # Check what's available after execution
    local_vars = locals()
    print(f"DependencyManager available: {'DependencyManager' in local_vars}")
    print(f"get_dependency_manager available: {'get_dependency_manager' in local_vars}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()
