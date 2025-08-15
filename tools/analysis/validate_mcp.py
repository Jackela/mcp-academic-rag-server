#!/usr/bin/env python3
"""
MCP Academic RAG Server - Validation Script
Quick validation of MCP setup and configuration
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        return False, f"Python {version.major}.{version.minor}.{version.micro}"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"

def check_uvx():
    """Check if uvx is available"""
    try:
        result = subprocess.run(['uvx', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, "uvx not found"
    except FileNotFoundError:
        return False, "uvx not installed"

def check_api_key():
    """Check if OpenAI API key is set and valid format"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return False, "OPENAI_API_KEY not set"
    if not api_key.startswith('sk-'):
        return False, f"Invalid format: {api_key[:10]}..."
    if len(api_key) < 20:
        return False, f"Too short: {api_key[:10]}..."
    return True, f"Valid format: {api_key[:8]}..."

def check_mcp_packages():
    """Check if MCP packages are available"""
    try:
        import mcp
        return True, f"MCP version available"
    except ImportError:
        return False, "MCP package not installed"

def check_project_structure():
    """Check if project files exist"""
    required_files = [
        'mcp_server.py',
        'mcp_server_secure.py',
        'config/config.json',
        'pyproject.toml'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        return False, f"Missing files: {', '.join(missing)}"
    return True, "All required files present"

def test_installation():
    """Test if the package can be installed and run"""
    try:
        # Test installation
        result = subprocess.run(['uvx', 'install', '.'], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode != 0:
            return False, f"Installation failed: {result.stderr}"
        
        # Test validation
        test_env = {**os.environ, 'OPENAI_API_KEY': 'sk-test123456789012345678'}
        result = subprocess.run(['uvx', 'run', 'mcp-academic-rag-server', '--validate-only'],
                              capture_output=True, text=True, env=test_env)
        
        if "Environment validation completed" in result.stderr:
            return True, "Package installation and validation successful"
        else:
            return False, f"Validation test failed: {result.stderr}"
            
    except Exception as e:
        return False, f"Test failed: {str(e)}"

def suggest_claude_config():
    """Generate Claude Desktop configuration suggestion"""
    api_key = os.environ.get('OPENAI_API_KEY', 'sk-your-api-key-here')
    config = {
        "mcpServers": {
            "academic-rag": {
                "command": "uvx",
                "args": ["run", "mcp-academic-rag-server"],
                "env": {
                    "OPENAI_API_KEY": api_key,
                    "DATA_PATH": "./data",
                    "LOG_LEVEL": "INFO"
                }
            }
        }
    }
    return json.dumps(config, indent=2)

def main():
    """Main validation function"""
    print("🔍 MCP Academic RAG Server - Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("uvx Package Manager", check_uvx),
        ("OpenAI API Key", check_api_key),
        ("MCP Packages", check_mcp_packages),
        ("Project Structure", check_project_structure),
        ("Installation Test", test_installation)
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        try:
            passed, message = check_func()
            status = "✅" if passed else "❌"
            print(f"{status} {name}: {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All checks passed! Your MCP server is ready.")
        print("\n📋 Claude Desktop Configuration:")
        print(suggest_claude_config())
        print(f"\n💡 Add this to your claude_desktop_config.json")
    else:
        print("⚠️  Some checks failed. Please resolve the issues above.")
        print("\n🔧 Common solutions:")
        print("  - Install uvx: pip install uvx")
        print("  - Set API key: export OPENAI_API_KEY=sk-your-key")
        print("  - Install MCP: pip install mcp")
    
    print(f"\n📁 Current directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.executable}")

if __name__ == "__main__":
    main()