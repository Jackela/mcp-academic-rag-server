#!/usr/bin/env python3
"""
MCP Inspector 模拟测试 - 验证RAG效果
模拟 MCP Inspector 的调试功能，对比简单模式vs完整RAG模式
"""

import asyncio
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.getcwd())

from mcp_server_standalone import MCPServer  # noqa: E402 - import follows source-script or runtime bootstrap.


class MCPInspectorSimulator:
    """模拟MCP Inspector的测试功能"""

    def __init__(self):
        self.server = MCPServer()
        self.test_results = []

    async def simulate_inspector_test(self):
        """模拟MCP Inspector的完整测试流程"""
        print("🔍 MCP Inspector 模拟测试")
        print("=" * 80)

        # 1. 连接测试
        await self._test_connection()

        # 2. 工具探索
        await self._explore_tools()

        # 3. 对比测试：简单模式 vs 智能RAG模式
        await self._compare_retrieval_modes()

        # 4. 性能测试
        await self._performance_test()

        # 5. 生成测试报告
        self._generate_report()

    async def _test_connection(self):
        """测试连接"""
        print("\n🔌 1. 连接测试")
        print("-" * 40)

        result = await self.server.handle_test_connection(1, {"message": "Inspector测试"})
        response = result["result"]["content"][0]["text"]

        # 检查RAG状态
        rag_available = "✅ 可用 (Sentence-BERT + OpenAI)" in response

        test_result = {
            "test": "connection",
            "status": "✅ 成功" if rag_available else "⚠️ 部分功能",
            "rag_mode": "完整RAG" if rag_available else "简单模式",
            "details": response[:200] + "..." if len(response) > 200 else response,
        }

        self.test_results.append(test_result)
        print(f"连接状态: {test_result['status']}")
        print(f"RAG模式: {test_result['rag_mode']}")

    async def _explore_tools(self):
        """探索可用工具"""
        print("\n🛠️ 2. 工具探索")
        print("-" * 40)

        # 模拟检查可用的MCP工具
        available_tools = ["test_connection", "process_document", "query_documents", "list_documents"]

        print("可用工具:")
        for tool in available_tools:
            print(f"  • {tool}")

        # 测试文档处理工具
        doc_result = await self.server.handle_process_document(2, {"file_path": "test-documents/machine-learning.txt"})

        processing_success = "error" not in doc_result

        test_result = {
            "test": "tool_exploration",
            "status": "✅ 成功" if processing_success else "❌ 失败",
            "tools_available": len(available_tools),
            "document_processing": "成功" if processing_success else "失败",
        }

        self.test_results.append(test_result)
        print(f"工具测试: {test_result['status']}")

    async def _compare_retrieval_modes(self):
        """对比不同检索模式的效果"""
        print("\n⚖️ 3. 检索模式对比测试")
        print("-" * 40)

        test_queries = [
            {
                "query": "What is machine learning?",
                "expected_concepts": ["artificial intelligence", "algorithms", "data", "patterns"],
            },
            {
                "query": "How do neural networks work?",
                "expected_concepts": ["layers", "neurons", "weights", "training"],
            },
        ]

        mode_comparison = []

        for query_info in test_queries:
            query = query_info["query"]
            print(f"\n🔍 查询: {query}")

            # 执行查询
            result = await self.server.handle_query_documents(3, {"query": query, "top_k": 2})

            if "error" not in result:
                response = result["result"]["content"][0]["text"]

                # 检查使用的模式
                if "🤖 智能RAG查询结果:" in response:
                    mode = "智能RAG"
                    quality = "高"
                elif "🔍 关键词匹配查询结果:" in response:
                    mode = "简单模式"
                    quality = "中"
                else:
                    mode = "未知"
                    quality = "低"

                # 评估回答质量
                response_quality = self._evaluate_response_quality(response, query_info["expected_concepts"])

                comparison_result = {
                    "query": query,
                    "mode": mode,
                    "quality": quality,
                    "response_length": len(response),
                    "concept_coverage": response_quality,
                    "status": "✅ 成功",
                }

                print(f"  模式: {mode}")
                print(f"  质量: {quality}")
                print(f"  概念覆盖: {response_quality}%")

            else:
                comparison_result = {
                    "query": query,
                    "mode": "错误",
                    "quality": "低",
                    "status": "❌ 失败",
                    "error": result["error"]["message"],
                }
                print("  状态: 查询失败")

            mode_comparison.append(comparison_result)

        test_result = {
            "test": "mode_comparison",
            "comparisons": mode_comparison,
            "intelligent_rag_success_rate": sum(1 for c in mode_comparison if c["mode"] == "智能RAG")
            / len(mode_comparison)
            * 100,
        }

        self.test_results.append(test_result)

        print(f"\n智能RAG成功率: {test_result['intelligent_rag_success_rate']:.1f}%")

    def _evaluate_response_quality(self, response, expected_concepts):
        """评估回答质量"""
        found_concepts = 0
        for concept in expected_concepts:
            if concept.lower() in response.lower():
                found_concepts += 1

        return (found_concepts / len(expected_concepts)) * 100 if expected_concepts else 0

    async def _performance_test(self):
        """性能测试"""
        print("\n⚡ 4. 性能测试")
        print("-" * 40)

        import time

        # 测试查询响应时间
        start_time = time.time()
        result = await self.server.handle_query_documents(4, {"query": "What is deep learning?", "top_k": 3})
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # 转换为毫秒

        performance_result = {
            "test": "performance",
            "response_time_ms": response_time,
            "status": "✅ 优秀" if response_time < 5000 else ("⚠️ 可接受" if response_time < 10000 else "❌ 慢"),
            "openai_calls": "成功" if "error" not in result else "失败",
        }

        self.test_results.append(performance_result)

        print(f"查询响应时间: {response_time:.0f}ms")
        print(f"性能评级: {performance_result['status']}")

    def _generate_report(self):
        """生成测试报告"""
        print("\n📊 5. MCP Inspector 测试报告")
        print("=" * 80)

        # 统计结果
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if "✅" in str(r.get("status", "")))

        print(f"总测试数: {total_tests}")
        print(f"成功测试: {successful_tests}")
        print(f"成功率: {successful_tests/total_tests*100:.1f}%")

        print("\n详细结果:")
        for i, result in enumerate(self.test_results, 1):
            print(f"\n{i}. {result['test'].replace('_', ' ').title()}")
            print(f"   状态: {result.get('status', 'N/A')}")

            if result["test"] == "connection":
                print(f"   RAG模式: {result.get('rag_mode', 'N/A')}")
            elif result["test"] == "mode_comparison":
                print(f"   智能RAG成功率: {result.get('intelligent_rag_success_rate', 0):.1f}%")
            elif result["test"] == "performance":
                print(f"   响应时间: {result.get('response_time_ms', 0):.0f}ms")

        # 总体评估
        print("\n🎯 总体评估:")
        if successful_tests == total_tests:
            print("🟢 优秀 - 所有测试通过，RAG系统运行完美")
        elif successful_tests >= total_tests * 0.8:
            print("🟡 良好 - 大多数测试通过，系统基本正常")
        else:
            print("🔴 需要改进 - 多个测试失败，系统存在问题")

        # 保存详细报告
        with open("mcp_inspector_test_report.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)

        print("\n📄 详细报告已保存到: mcp_inspector_test_report.json")


async def main():
    """主函数"""
    inspector = MCPInspectorSimulator()
    await inspector.simulate_inspector_test()


if __name__ == "__main__":
    asyncio.run(main())
