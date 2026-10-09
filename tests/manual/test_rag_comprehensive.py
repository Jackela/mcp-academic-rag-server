#!/usr/bin/env python3
"""
Comprehensive RAG Pipeline Test
验证完整的RAG功能：文档处理 -> 嵌入生成 -> 语义检索 -> LLM生成答案
"""

import asyncio
import os
import sys

# 添加项目路径
sys.path.insert(0, os.getcwd())

# 导入服务器
from mcp_server_standalone import MCPServer  # noqa: E402 - import follows source-script or runtime bootstrap.


async def test_rag_pipeline():
    """测试完整RAG管道功能"""
    print("🧪 开始RAG管道综合测试")
    print("=" * 80)

    # 初始化服务器
    server = MCPServer()

    # 测试1: 检查RAG状态
    print("\n🔍 测试1: 检查RAG初始化状态")
    connection_result = await server.handle_test_connection(1, {"message": "RAG测试"})
    response_text = connection_result["result"]["content"][0]["text"]
    print("Response:", response_text)

    rag_available = "✅ 可用 (Sentence-BERT + OpenAI)" in response_text
    print(f"RAG状态: {'✅ 完整RAG可用' if rag_available else '❌ 仅关键词模式'}")

    if not rag_available:
        print("⚠️ 完整RAG不可用，检查API密钥和依赖")
        return

    # 测试2: 处理多个文档
    print("\n📄 测试2: 处理测试文档")
    test_docs = [
        "test-documents/machine-learning.txt",
        "test-documents/neural-networks.txt",
        "test-documents/nlp-research.txt",
    ]

    for doc_path in test_docs:
        print(f"\n处理文档: {doc_path}")
        result = await server.handle_process_document(2, {"file_path": doc_path})

        if "error" in result:
            print(f"❌ 处理失败: {result['error']['message']}")
            continue

        response = result["result"]["content"][0]["text"]
        print("处理结果:")
        print(response[:500] + "..." if len(response) > 500 else response)

        # 检查是否使用完整RAG
        if "RAG处理状态: completed" in response:
            print("✅ 使用完整RAG管道处理")
        else:
            print("❌ 未使用完整RAG管道")

    # 测试3: 智能查询测试
    print("\n🤖 测试3: 智能RAG查询")

    test_queries = [
        {"query": "What is machine learning and how is it used in research?", "expected": "应该提供ML的定义和研究应用"},
        {"query": "Explain neural networks and deep learning", "expected": "应该解释神经网络架构和深度学习"},
        {"query": "How does NLP help in academic research?", "expected": "应该说明NLP在学术研究中的应用"},
        {
            "query": "What are the differences between supervised and unsupervised learning?",
            "expected": "应该对比不同学习类型",
        },
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {test['query']}")
        print(f"期望: {test['expected']}")

        result = await server.handle_query_documents(3, {"query": test["query"], "top_k": 3})

        if "error" in result:
            print(f"❌ 查询失败: {result['error']['message']}")
            continue

        response = result["result"]["content"][0]["text"]
        print("\n查询结果:")
        print(response)

        # 检查是否使用智能RAG
        if "🤖 智能RAG查询结果:" in response:
            print("✅ 使用智能RAG查询")
        elif "🔍 关键词匹配查询结果:" in response:
            print("⚠️ 回退到关键词匹配")
        else:
            print("❓ 未知查询模式")

        print("-" * 60)

    # 测试4: 检查文档列表
    print("\n📚 测试4: 查看处理的文档")
    docs_result = await server.handle_list_documents(4, {})
    docs_response = docs_result["result"]["content"][0]["text"]
    print(docs_response)

    print("\n🎉 RAG管道测试完成!")


def main():
    """主函数"""
    asyncio.run(test_rag_pipeline())


if __name__ == "__main__":
    main()
