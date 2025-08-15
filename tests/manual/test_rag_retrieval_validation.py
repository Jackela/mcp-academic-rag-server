#!/usr/bin/env python3
"""
RAG检索验证测试 - 验证是否真正从文档中检索内容
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.getcwd())

from mcp_server_standalone import MCPServer

async def test_rag_retrieval():
    """验证RAG是否真正从文档检索内容"""
    print("🔍 RAG检索验证测试")
    print("="*80)
    
    server = MCPServer()
    
    # 1. 处理包含特定信息的测试文档
    print("\n📄 1. 处理测试文档")
    await server.handle_process_document(1, {"file_path": "test-documents/machine-learning.txt"})
    await server.handle_process_document(2, {"file_path": "test-documents/neural-networks.txt"})
    await server.handle_process_document(3, {"file_path": "test-documents/nlp-research.txt"})
    
    # 2. 测试特定文档内容查询
    print("\n🔍 2. 测试特定文档内容查询")
    
    # 查询应该能在文档中找到的特定内容
    specific_queries = [
        "What are the core NLP technologies mentioned?",  # 应该找到 tokenization, POS tagging等
        "What machine learning algorithms are mentioned?",  # 应该找到 decision trees, SVM等
        "What are the applications of neural networks?",   # 应该找到具体应用
    ]
    
    for i, query in enumerate(specific_queries):
        print(f"\n查询 {i+1}: {query}")
        
        result = await server.handle_query_documents(10+i, {"query": query, "top_k": 3})
        
        if "error" not in result:
            response = result["result"]["content"][0]["text"]
            
            # 检查是否真的使用了文档检索
            print("响应长度:", len(response))
            
            if "🤖 智能RAG查询结果:" in response:
                print("✅ 使用智能RAG模式")
                
                # 检查是否包含文档引用
                if "文档ID:" in response or "来源：" in response:
                    print("✅ 包含文档引用")
                else:
                    print("⚠️ 没有文档引用")
                
                # 检查相关文档片段部分
                if "📁 相关文档片段 (前0个)" in response:
                    print("❌ 没有检索到文档片段!")
                elif "📁 相关文档片段" in response:
                    # 提取文档片段数量
                    if "(前" in response and "个)" in response:
                        start = response.find("(前") + 2
                        end = response.find("个)", start)
                        count = response[start:end]
                        print(f"✅ 检索到 {count} 个文档片段")
                
            else:
                print("❌ 没有使用智能RAG")
        
        print("-" * 40)
    
    # 3. 测试不在文档中的内容
    print("\n🔍 3. 测试不在文档中的内容")
    
    unknown_query = "What is quantum computing in blockchain applications?"
    print(f"查询: {unknown_query}")
    
    result = await server.handle_query_documents(20, {"query": unknown_query, "top_k": 3})
    
    if "error" not in result:
        response = result["result"]["content"][0]["text"]
        
        if "📁 相关文档片段 (前0个)" in response:
            print("✅ 正确：没有检索到相关文档片段 (内容不在文档中)")
        elif "📁 相关文档片段" in response and "(前" in response:
            print("❌ 错误：检索到了不相关的文档片段")
    
    print("\n🎯 结论:")
    print("如果所有查询都显示'检索到0个文档片段'，")
    print("说明检索器存在问题，回答来自模型参数化知识而非文档。")

if __name__ == "__main__":
    asyncio.run(test_rag_retrieval())