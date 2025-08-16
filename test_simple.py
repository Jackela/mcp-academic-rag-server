#!/usr/bin/env python3
"""
Simple standalone test to verify CI/CD functionality without complex conftest.py
"""

import sys
import os
import tempfile
import json

def test_python_version():
    """Test that we're running a supported Python version."""
    assert sys.version_info >= (3, 9), f"Python 3.9+ required, got {sys.version_info}"
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")

def test_basic_imports():
    """Test basic imports work."""
    import json
    import os
    import sys
    import tempfile
    import logging
    assert json is not None
    assert os is not None
    assert sys is not None
    assert tempfile is not None
    assert logging is not None
    print("✅ Basic imports successful")

def test_file_operations():
    """Test basic file operations."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("test content")
        temp_path = temp_file.name
    
    # Verify file exists and has content
    assert os.path.exists(temp_path)
    
    with open(temp_path, 'r') as f:
        content = f.read()
        assert content == "test content"
    
    # Clean up
    os.unlink(temp_path)
    print("✅ File operations successful")

def test_json_handling():
    """Test JSON operations."""
    test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    
    # Test serialization
    json_string = json.dumps(test_data)
    assert isinstance(json_string, str)
    
    # Test deserialization
    parsed_data = json.loads(json_string)
    assert parsed_data == test_data
    print("✅ JSON handling successful")

def test_environment_access():
    """Test environment variable access."""
    # Test that we can access environment variables
    python_path = os.environ.get('PYTHONPATH', '')
    assert isinstance(python_path, str)
    
    # Test setting and getting env vars
    test_key = 'TEST_ENV_VAR_12345'
    test_value = 'test_value'
    os.environ[test_key] = test_value
    assert os.environ.get(test_key) == test_value
    
    # Clean up
    del os.environ[test_key]
    print("✅ Environment access successful")

def test_optional_dependencies():
    """Test availability of optional dependencies."""
    try:
        import anthropic
        print("✅ Anthropic package available")
    except ImportError:
        print("⚠️ Anthropic package not available")
    
    try:
        import openai
        print("✅ OpenAI package available")
    except ImportError:
        print("⚠️ OpenAI package not available")
    
    try:
        import pytest
        print("✅ Pytest package available")
    except ImportError:
        print("❌ Pytest package not available")
        raise

if __name__ == "__main__":
    print("🚀 Running simple tests...")
    
    try:
        test_python_version()
        test_basic_imports()
        test_file_operations()
        test_json_handling()
        test_environment_access()
        test_optional_dependencies()
        
        print("\n🎉 All simple tests passed!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)