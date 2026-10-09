#!/usr/bin/env python3
"""
单个RAG查询测试 - 验证完整管道
"""

import asyncio
import os
import sys

# 添加项目路径
sys.path.insert(0, os.getcwd())

from mcp_server_standalone import MCPServer


async def test_single_rag_query():
    """测试单个RAG查询以验证完整功能"""
    print("🧪 测试完整RAG查询管道")
    print("=" * 60)

    # 初始化服务器
    server = MCPServer()

    # 1. 检查RAG状态
    print("1. 检查RAG状态...")
    connection_result = await server.handle_test_connection(1, {"message": "RAG完整测试"})
    print(
        f"RAG状态: {'可用' if '✅ 可用 (Sentence-BERT + OpenAI)' in connection_result['result']['content'][0]['text'] else '不可用'}"
    )

    # 2. 处理测试文档
    print("\n2. 处理测试文档...")
    doc_result = await server.handle_process_document(2, {"file_path": "test-documents/machine-learning.txt"})
    if "error" not in doc_result:
        print("✅ 文档处理成功")
    else:
        print(f"❌ 文档处理失败: {doc_result['error']['message']}")
        return

    # 3. 测试智能RAG查询
    print("\n3. 测试智能RAG查询...")
    query = "What is machine learning and how is it used in research?"
    print(f"查询: {query}")

    query_result = await server.handle_query_documents(3, {"query": query, "top_k": 2})

    if "error" not in query_result:
        response = query_result["result"]["content"][0]["text"]
        print("\n查询响应:")
        print("-" * 60)
        print(response)
        print("-" * 60)

        # 检查是否使用智能RAG
        if "🤖 智能RAG查询结果:" in response:
            print("\n✅ 成功使用完整RAG管道！")
            print("🎯 包含: 嵌入检索 + OpenAI生成")
        elif "🔍 关键词匹配查询结果:" in response:
            print("\n⚠️ 使用了简单模式")
        else:
            print("\n❓ 未知查询模式")
    else:
        print(f"❌ 查询失败: {query_result['error']['message']}")


if __name__ == "__main__":
    asyncio.run(test_single_rag_query())
