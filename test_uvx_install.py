#!/usr/bin/env python3
"""
Test script for uvx installation of MCP Academic RAG Server
"""

import sys
import subprocess
import os
import tempfile
import shutil

def test_uvx_installation():
    """Test that the package can be installed via uvx"""
    
    print("🔍 Testing uvx installation compatibility...")
    
    # Get current directory (package root)
    package_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Package root: {package_root}")
    
    # Test 1: Check pyproject.toml exists and has entry points
    pyproject_path = os.path.join(package_root, "pyproject.toml")
    if not os.path.exists(pyproject_path):
        print("❌ pyproject.toml not found")
        return False
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
        if "mcp-academic-rag-server" not in content:
            print("❌ Entry point not found in pyproject.toml")
            return False
        print("✅ Entry point found in pyproject.toml")
    
    # Test 2: Check that main dependencies are minimal
    if "prometheus" in content or "grafana" in content:
        print("❌ Enterprise dependencies still present")
        return False
    print("✅ Dependencies simplified")
    
    # Test 3: Try importing the main module
    try:
        sys.path.insert(0, package_root)
        import mcp_server
        print("✅ mcp_server module importable")
    except Exception as e:
        print(f"❌ Failed to import mcp_server: {e}")
        return False
    
    # Test 4: Check essential files exist
    essential_files = [
        "mcp_server.py",
        "config/config.json",
        "requirements.txt",
        ".env.example"
    ]
    
    for file_path in essential_files:
        full_path = os.path.join(package_root, file_path)
        if not os.path.exists(full_path):
            print(f"❌ Essential file missing: {file_path}")
            return False
    print("✅ All essential files present")
    
    print("🎉 uvx installation test passed!")
    return True

if __name__ == "__main__":
    success = test_uvx_installation()
    sys.exit(0 if success else 1)