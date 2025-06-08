#!/usr/bin/env python3
"""Minimal test to check RAG module import issues"""

print("Starting minimal RAG import test...")

try:
    print("1. Testing basic imports...")
    import os
    import sys
    from typing import List, Optional
    print("   ✓ Basic imports successful")
    
    print("2. Testing intv package...")
    import intv
    print("   ✓ intv package imported")
    
    print("3. Testing intv.utils...")
    try:
        from intv.utils import is_valid_filetype
        print("   ✓ intv.utils imported")
    except Exception as e:
        print(f"   ⚠ intv.utils failed: {e}")
    
    print("4. Testing intv.rag...")
    try:
        import intv.rag as rag_module
        print("   ✓ intv.rag imported successfully")
        print(f"   ✓ Available functions: {[name for name in dir(rag_module) if not name.startswith('_')][:5]}")
    except Exception as e:
        print(f"   ✗ intv.rag failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n✅ Minimal test completed")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
