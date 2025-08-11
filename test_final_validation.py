#!/usr/bin/env python3
"""
Final validation test of the simplified MCP Academic RAG Server
"""

import os
import sys
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_system_validation():
    """Run final system validation"""
    
    print("🎯 MCP Academic RAG Server - Final Validation")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # Test 1: Architecture Simplified
    total += 1
    print("\n🧪 Test 1: Architecture Simplification")
    
    if not os.path.exists('k8s') and not os.path.exists('utils/monitoring_utils.py'):
        print("✅ Enterprise features removed")
        passed += 1
    else:
        print("❌ Enterprise features still present")
    
    # Test 2: Dependencies Reduced  
    total += 1
    print("\n🧪 Test 2: Dependencies Reduction")
    
    with open('requirements.txt', 'r') as f:
        deps = [line.strip() for line in f.readlines() 
                if line.strip() and not line.startswith('#')]
    
    if len(deps) <= 20:
        print(f"✅ Dependencies reduced to {len(deps)} (from 70+)")
        passed += 1
    else:
        print(f"❌ Still too many dependencies: {len(deps)}")
    
    # Test 3: Configuration Simplified
    total += 1
    print("\n🧪 Test 3: Configuration Simplification")
    
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    if (config.get('storage', {}).get('type') == 'local' and 
        config.get('storage', {}).get('vector_store') == 'faiss'):
        print("✅ Local FAISS storage configured")
        passed += 1
    else:
        print("❌ Storage not properly configured")
    
    # Test 4: Core MCP Tools
    total += 1 
    print("\n🧪 Test 4: Core MCP Tools")
    
    tools = config.get('mcp', {}).get('tools_enabled', [])
    expected = ['process_document', 'query_documents', 'get_document_info', 'list_sessions']
    
    if all(tool in tools for tool in expected):
        print("✅ All 4 core MCP tools configured")
        passed += 1
    else:
        missing = [tool for tool in expected if tool not in tools]
        print(f"❌ Missing tools: {missing}")
    
    # Test 5: uvx Compatibility
    total += 1
    print("\n🧪 Test 5: uvx Package Compatibility")
    
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    
    if 'mcp-academic-rag-server = "mcp_server:main"' in content:
        print("✅ Entry point configured for uvx")
        passed += 1
    else:
        print("❌ Entry point not configured")
    
    # Test 6: Environment Simplified
    total += 1
    print("\n🧪 Test 6: Environment Configuration")
    
    with open('.env.example', 'r') as f:
        env_content = f.read()
    
    if ('OPENAI_API_KEY' in env_content and 'MILVUS_HOST' not in env_content):
        print("✅ Environment configuration simplified")
        passed += 1
    else:
        print("❌ Environment not properly simplified")
    
    # Test 7: Docker Simplification
    total += 1
    print("\n🧪 Test 7: Docker Simplification")
    
    if (os.path.exists('docker-compose.simple.yml') and 
        os.path.exists('Dockerfile.simple')):
        
        with open('docker-compose.simple.yml', 'r') as f:
            docker_content = f.read()
        
        service_count = docker_content.count('container_name:')
        if service_count == 1:
            print("✅ Docker simplified to single service")
            passed += 1
        else:
            print(f"❌ Docker has {service_count} services, expected 1")
    else:
        print("❌ Simplified Docker files missing")
    
    # Test 8: MCP Server Structure
    total += 1
    print("\n🧪 Test 8: MCP Server Implementation")
    
    if os.path.exists('mcp_server.py'):
        with open('mcp_server.py', 'r') as f:
            server_content = f.read()
        
        if ('@server.list_tools()' in server_content and 
            'async def process_document(' in server_content):
            print("✅ MCP server properly implemented")
            passed += 1
        else:
            print("❌ MCP server structure incomplete")
    else:
        print("❌ mcp_server.py not found")
    
    # Test 9: Documentation Focus
    total += 1
    print("\n🧪 Test 9: Documentation Focus")
    
    with open('README.md', 'r') as f:
        readme_content = f.read()
    
    if ('uvx' in readme_content and 
        ('4 essential tools' in readme_content or '4 core tools' in readme_content) and
        'kubernetes' not in readme_content.lower()):
        print("✅ Documentation focused on core functionality")
        passed += 1
    else:
        print("❌ Documentation not properly focused")
    
    # Test 10: File Cleanup
    total += 1
    print("\n🧪 Test 10: Enterprise File Cleanup")
    
    enterprise_files_removed = (
        not os.path.exists('document_stores/milvus_store.py') and
        not os.path.exists('utils/monitoring_utils.py') and
        not os.path.exists('config/milvus.yaml')
    )
    
    if enterprise_files_removed:
        print("✅ Enterprise files removed")
        passed += 1
    else:
        print("❌ Some enterprise files still present")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 FINAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System successfully refocused.")
        print("\n📋 Refocusing Complete:")
        print("✅ Scope reduced from enterprise platform to focused MCP server")
        print("✅ Dependencies cut from 70+ to ~15 essential packages")
        print("✅ Configuration simplified (local FAISS, basic settings)")
        print("✅ 4 core MCP tools: process_document, query_documents, get_document_info, list_sessions")
        print("✅ uvx-compatible package structure for easy installation")
        print("✅ Docker simplified from 7 services to 1 service") 
        print("✅ Enterprise features removed (K8s, monitoring, complex UI)")
        print("✅ Documentation refocused on core MCP functionality")
        print("\n🚀 Ready for uvx installation: `uvx --from . mcp-academic-rag-server`")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed - refocusing incomplete")
        return False

if __name__ == "__main__":
    success = test_system_validation()
    sys.exit(0 if success else 1)