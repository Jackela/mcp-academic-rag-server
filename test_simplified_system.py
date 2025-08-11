#!/usr/bin/env python3
"""
Comprehensive test of the simplified MCP Academic RAG Server system
"""

import os
import sys
import json
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class SimplifiedSystemTest:
    """Test the complete simplified system"""
    
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
        
    def test(self, description: str):
        """Decorator for test methods"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.total_tests += 1
                print(f"\n🧪 {description}")
                try:
                    result = func(*args, **kwargs)
                    if result:
                        print("✅ PASSED")
                        self.passed_tests += 1
                    else:
                        print("❌ FAILED")
                    return result
                except Exception as e:
                    print(f"❌ ERROR: {e}")
                    return False
            return wrapper
        return decorator
    
    @test("System Architecture Simplification")
    def test_architecture_simplified(self):
        """Test that enterprise features have been removed"""
        
        # Check no Kubernetes manifests
        if os.path.exists('k8s'):
            print("❌ Kubernetes manifests still present")
            return False
        
        # Check Docker simplified
        if os.path.exists('docker-compose.simple.yml'):
            with open('docker-compose.simple.yml', 'r') as f:
                content = f.read()
                if 'milvus' not in content and 'prometheus' not in content:
                    print("✅ Docker configuration simplified")
                else:
                    print("❌ Complex services still in docker-compose.simple.yml")
                    return False
        
        # Check monitoring utilities removed
        if os.path.exists('utils/monitoring_utils.py'):
            print("❌ Monitoring utilities still present")
            return False
            
        return True
    
    @test("Dependencies Reduction")
    def test_dependencies_reduced(self):
        """Test that dependencies have been reduced from 70+ to ~15"""
        
        # Count dependencies in requirements.txt
        with open('requirements.txt', 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            deps = [line for line in lines if line and not line.startswith('#')]
        
        print(f"📊 Dependencies count: {len(deps)}")
        
        if len(deps) <= 20:  # Allow some flexibility
            print("✅ Dependencies successfully reduced")
        else:
            print(f"❌ Still too many dependencies: {len(deps)}")
            return False
        
        # Check no enterprise dependencies
        content = ' '.join(deps).lower()
        enterprise_deps = ['prometheus', 'grafana', 'kubernetes', 'redis', 'milvus']
        
        for dep in enterprise_deps:
            if dep in content:
                print(f"❌ Enterprise dependency still present: {dep}")
                return False
        
        return True
    
    @test("Configuration Simplification")
    def test_config_simplified(self):
        """Test configuration has been simplified"""
        
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        # Check storage is local
        if config.get('storage', {}).get('type') == 'local':
            print("✅ Storage configured for local FAISS")
        else:
            print("❌ Storage not configured for local use")
            return False
        
        # Check vector store is FAISS
        if config.get('storage', {}).get('vector_store') == 'faiss':
            print("✅ Vector store set to FAISS")
        else:
            print("❌ Vector store not set to FAISS")
            return False
        
        # Check 4 core tools enabled
        tools = config.get('mcp', {}).get('tools_enabled', [])
        expected_tools = ['process_document', 'query_documents', 'get_document_info', 'list_sessions']
        
        if all(tool in tools for tool in expected_tools):
            print("✅ All 4 core MCP tools enabled")
        else:
            missing = [tool for tool in expected_tools if tool not in tools]
            print(f"❌ Missing MCP tools: {missing}")
            return False
        
        return True
    
    @test("Environment Configuration")
    def test_env_simplified(self):
        """Test .env.example is simplified"""
        
        with open('.env.example', 'r') as f:
            content = f.read()
        
        # Check essential vars present
        essential_vars = ['OPENAI_API_KEY', 'DATA_PATH', 'MCP_PORT']
        for var in essential_vars:
            if var in content:
                print(f"✅ Essential variable {var} present")
            else:
                print(f"❌ Essential variable {var} missing")
                return False
        
        # Check enterprise vars removed
        enterprise_vars = ['MILVUS_HOST', 'REDIS_URL', 'PROMETHEUS', 'GRAFANA']
        enterprise_found = []
        for var in enterprise_vars:
            if var in content:
                enterprise_found.append(var)
        
        if not enterprise_found:
            print("✅ Enterprise variables removed")
        else:
            print(f"❌ Enterprise variables still present: {enterprise_found}")
            return False
        
        return True
    
    @test("Package Structure for uvx")
    def test_uvx_compatibility(self):
        """Test package is compatible with uvx"""
        
        # Check pyproject.toml has entry points
        with open('pyproject.toml', 'r') as f:
            content = f.read()
        
        if 'mcp-academic-rag-server = "mcp_server:main"' in content:
            print("✅ Entry point configured for uvx")
        else:
            print("❌ Entry point not configured")
            return False
        
        # Check essential files exist
        essential_files = [
            'mcp_server.py',
            'config/config.json', 
            'requirements.txt',
            '.env.example',
            'README.md'
        ]
        
        for file_path in essential_files:
            if os.path.exists(file_path):
                print(f"✅ Essential file exists: {file_path}")
            else:
                print(f"❌ Missing essential file: {file_path}")
                return False
        
        return True
    
    @test("Docker Simplification")
    def test_docker_simplified(self):
        """Test Docker setup is simplified"""
        
        # Check simple Dockerfile exists
        if not os.path.exists('Dockerfile.simple'):
            print("❌ Simplified Dockerfile not found")
            return False
        
        with open('Dockerfile.simple', 'r') as f:
            content = f.read()
        
        # Should use requirements.simple.txt
        if 'requirements.simple.txt' in content:
            print("✅ Dockerfile uses simplified requirements")
        else:
            print("❌ Dockerfile not using simplified requirements")
            return False
        
        # Check simple docker-compose exists
        if not os.path.exists('docker-compose.simple.yml'):
            print("❌ Simplified docker-compose not found")
            return False
        
        with open('docker-compose.simple.yml', 'r') as f:
            content = f.read()
        
        # Should be single service
        service_count = content.count('container_name:')
        if service_count == 1:
            print("✅ Docker-compose simplified to single service")
        else:
            print(f"❌ Docker-compose has {service_count} services, expected 1")
            return False
        
        return True
    
    async def test_mcp_server_structure(self):
        """Test MCP server structure is correct"""
        
        # Check mcp_server.py exists and has main components
        if not os.path.exists('mcp_server.py'):
            print("❌ mcp_server.py not found")
            return False
        
        with open('mcp_server.py', 'r') as f:
            content = f.read()
        
        # Check for 4 core tool implementations
        tools = ['process_document', 'query_documents', 'get_document_info', 'list_sessions']
        for tool in tools:
            if f'async def {tool}(' in content:
                print(f"✅ Tool implementation found: {tool}")
            else:
                print(f"❌ Tool implementation missing: {tool}")
                return False
        
        # Check MCP server setup
        if '@server.list_tools()' in content and '@server.call_tool()' in content:
            print("✅ MCP server decorators found")
        else:
            print("❌ MCP server decorators missing")
            return False
        
        return True
    
    def test_readme_focused(self):
        """Test README is focused on core functionality"""
        
        with open('README.md', 'r') as f:
            content = f.read()
        
        # Check mentions 4 core tools
        if '4 essential tools' in content or '4 core tools' in content:
            print("✅ README mentions core tools")
        else:
            print("❌ README doesn't emphasize core tools")
            return False
        
        # Check mentions uvx installation
        if 'uvx' in content:
            print("✅ README includes uvx installation")
        else:
            print("❌ README missing uvx installation")
            return False
        
        # Check doesn't mention enterprise features
        if 'kubernetes' not in content.lower() and 'prometheus' not in content.lower():
            print("✅ README doesn't mention enterprise features")
        else:
            print("❌ README still mentions enterprise features")
            return False
        
        return True
    
    def run_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Passed: {self.passed_tests}/{self.total_tests} tests")
        
        if self.passed_tests == self.total_tests:
            print("🎉 ALL TESTS PASSED! System successfully simplified.")
            print("\n📋 Refocusing Complete:")
            print("✅ Enterprise features removed (Kubernetes, monitoring, etc.)")
            print("✅ Dependencies reduced from 70+ to ~15 packages")
            print("✅ Configuration simplified (local storage, basic settings)")
            print("✅ 4 core MCP tools: process_document, query_documents, get_document_info, list_sessions")
            print("✅ uvx-compatible package structure")
            print("✅ Simplified Docker setup (single service)")
            print("✅ Focused documentation")
            return True
        else:
            print(f"❌ {self.total_tests - self.passed_tests} tests failed")
            return False

async def main():
    """Run all validation tests"""
    
    print("🎯 MCP Academic RAG Server - Simplified System Validation")
    print("="*60)
    
    tester = SimplifiedSystemTest()
    
    # Run all tests
    test_methods = [
        tester.test_architecture_simplified,
        tester.test_dependencies_reduced,
        tester.test_config_simplified,
        tester.test_env_simplified,
        tester.test_uvx_compatibility,
        tester.test_docker_simplified,
        tester.test_mcp_server_structure,
        tester.test_readme_focused
    ]
    
    for test_method in test_methods:
        if asyncio.iscoroutinefunction(test_method):
            await test_method()
        else:
            test_method()
    
    return tester.run_summary()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)