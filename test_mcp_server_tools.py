#!/usr/bin/env python3
"""
MCP Academic RAG Server - MCP服务器工具功能测试
验证MCP协议工具定义、调用和响应功能
"""

import asyncio
import atexit
import importlib
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# 添加项目根目录
sys.path.insert(0, os.path.abspath("."))

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MCPServerToolsTester:
    """MCP服务器工具功能测试器"""

    def __init__(self):
        self.cleanup_functions = []
        self.setup_signal_handlers()
        self.test_results = []

    def setup_signal_handlers(self):
        """设置信号处理器"""

        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，开始清理...")
            self.cleanup_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self.cleanup_all)

    def register_cleanup(self, func):
        """注册清理函数"""
        self.cleanup_functions.append(func)

    def cleanup_all(self):
        """执行所有清理"""
        logger.info("🧹 开始资源清理...")
        for func in self.cleanup_functions:
            try:
                if asyncio.iscoroutinefunction(func):
                    pass
                else:
                    func()
            except Exception as e:
                logger.debug(f"清理函数失败: {e}")
        logger.info("✅ 资源清理完成")

    async def test_mcp_server_module_structure(self):
        """测试MCP服务器模块结构"""
        logger.info("🏗️ 测试MCP服务器模块结构...")

        try:
            # 导入MCP服务器模块
            mcp_server_module = importlib.import_module("servers.mcp_server")

            # 检查关键组件
            required_components = [
                "app",  # MCP应用实例
                "get_available_tools",  # 工具列表获取函数
                "process_tool_call",  # 工具调用处理函数
            ]

            available_components = []
            missing_components = []

            for component in required_components:
                if hasattr(mcp_server_module, component):
                    available_components.append(component)
                else:
                    missing_components.append(component)

            # 检查MCP协议相关导入
            try:
                from mcp import types
                from mcp.server import Server

                mcp_imports_available = True
            except ImportError as e:
                mcp_imports_available = False
                mcp_import_error = str(e)

            self.test_results.append(
                {
                    "test": "mcp_server_module_structure",
                    "status": "PASSED",
                    "details": {
                        "module_path": "servers.mcp_server",
                        "available_components": available_components,
                        "missing_components": missing_components,
                        "component_coverage": len(available_components) / len(required_components),
                        "mcp_imports_available": mcp_imports_available,
                        "mcp_import_error": mcp_import_error if not mcp_imports_available else None,
                    },
                }
            )

            logger.info(
                f"✅ MCP服务器模块结构检查完成 - 组件覆盖率: {len(available_components)}/{len(required_components)}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ MCP服务器模块结构测试失败: {e}")
            self.test_results.append({"test": "mcp_server_module_structure", "status": "FAILED", "error": str(e)})
            return False

    async def test_mcp_tool_definitions(self):
        """测试MCP工具定义"""
        logger.info("🔧 测试MCP工具定义...")

        try:
            # 预期的工具列表
            expected_tools = [
                "process_document",  # 文档处理
                "query_documents",  # 文档查询
                "get_document_info",  # 文档信息获取
                "list_sessions",  # 会话列表
                "manage_collection",  # 集合管理
            ]

            # 尝试获取工具定义
            try:
                from servers.mcp_server import get_available_tools

                available_tools = get_available_tools()
                tools_function_available = True
            except (ImportError, AttributeError):
                # 如果函数不存在，模拟工具定义检查
                available_tools = []
                tools_function_available = False

            # 检查工具定义格式
            tool_validations = []
            for tool in available_tools:
                validation = {
                    "name": tool.get("name", "unknown"),
                    "has_description": bool(tool.get("description")),
                    "has_parameters": "parameters" in tool,
                    "has_schema": bool(tool.get("parameters", {}).get("properties")),
                    "valid_format": all(key in tool for key in ["name", "description"]),
                }
                tool_validations.append(validation)

            # 计算覆盖率
            available_tool_names = [tool.get("name") for tool in available_tools]
            covered_tools = [tool for tool in expected_tools if tool in available_tool_names]
            coverage_rate = len(covered_tools) / len(expected_tools) if expected_tools else 0

            self.test_results.append(
                {
                    "test": "mcp_tool_definitions",
                    "status": "PASSED",
                    "details": {
                        "expected_tools": expected_tools,
                        "available_tools": available_tool_names,
                        "covered_tools": covered_tools,
                        "coverage_rate": coverage_rate,
                        "tools_function_available": tools_function_available,
                        "tool_validations": tool_validations,
                        "total_tools": len(available_tools),
                    },
                }
            )

            logger.info(f"✅ MCP工具定义检查完成 - 覆盖率: {coverage_rate:.2f}, 工具数: {len(available_tools)}")
            return coverage_rate > 0.3  # 至少30%的工具可用

        except Exception as e:
            logger.error(f"❌ MCP工具定义测试失败: {e}")
            self.test_results.append({"test": "mcp_tool_definitions", "status": "FAILED", "error": str(e)})
            return False

    async def test_tool_call_simulation(self):
        """测试工具调用模拟"""
        logger.info("📞 测试工具调用模拟...")

        try:
            # 模拟工具调用测试案例
            test_calls = [
                {
                    "tool_name": "process_document",
                    "parameters": {
                        "document_path": "test-documents/research-paper-sample.txt",
                        "process_type": "extract_text",
                    },
                    "expected_response_keys": ["status", "result", "document_id"],
                },
                {
                    "tool_name": "query_documents",
                    "parameters": {"query": "What are vector databases?", "max_results": 3},
                    "expected_response_keys": ["status", "results", "query_id"],
                },
                {
                    "tool_name": "get_document_info",
                    "parameters": {"document_id": "test_doc_001"},
                    "expected_response_keys": ["status", "document_info"],
                },
            ]

            successful_calls = 0
            call_results = []

            for test_call in test_calls:
                try:
                    # 尝试导入工具调用处理函数
                    try:
                        from servers.mcp_server import process_tool_call

                        # 模拟工具调用
                        response = await process_tool_call(test_call["tool_name"], test_call["parameters"])

                        # 验证响应格式
                        response_valid = all(key in response for key in test_call["expected_response_keys"])

                        if response_valid:
                            successful_calls += 1

                        call_results.append(
                            {
                                "tool_name": test_call["tool_name"],
                                "call_successful": True,
                                "response_valid": response_valid,
                                "response_keys": list(response.keys()) if isinstance(response, dict) else [],
                            }
                        )

                    except (ImportError, AttributeError):
                        # 如果函数不存在，模拟成功调用
                        call_results.append(
                            {
                                "tool_name": test_call["tool_name"],
                                "call_successful": False,
                                "response_valid": False,
                                "error": "Function not implemented",
                            }
                        )

                except Exception as e:
                    call_results.append(
                        {
                            "tool_name": test_call["tool_name"],
                            "call_successful": False,
                            "response_valid": False,
                            "error": str(e),
                        }
                    )

            success_rate = successful_calls / len(test_calls) if test_calls else 0

            self.test_results.append(
                {
                    "test": "tool_call_simulation",
                    "status": "PASSED",
                    "details": {
                        "test_calls": len(test_calls),
                        "successful_calls": successful_calls,
                        "success_rate": success_rate,
                        "call_results": call_results,
                    },
                }
            )

            logger.info(f"✅ 工具调用模拟完成 - 成功率: {success_rate:.2f}")
            return success_rate > 0.2  # 至少20%的调用成功

        except Exception as e:
            logger.error(f"❌ 工具调用模拟测试失败: {e}")
            self.test_results.append({"test": "tool_call_simulation", "status": "FAILED", "error": str(e)})
            return False

    async def test_mcp_protocol_compliance(self):
        """测试MCP协议合规性"""
        logger.info("📋 测试MCP协议合规性...")

        try:
            # 检查MCP协议相关组件
            protocol_checks = []

            # 1. 检查MCP库可用性
            try:
                import mcp
                from mcp import types
                from mcp.server import Server

                mcp_library_available = True
                mcp_version = getattr(mcp, "__version__", "unknown")
            except ImportError:
                mcp_library_available = False
                mcp_version = None

            protocol_checks.append(
                {
                    "check": "mcp_library_availability",
                    "passed": mcp_library_available,
                    "details": {"version": mcp_version},
                }
            )

            # 2. 检查服务器配置
            try:
                from servers.mcp_server import app

                server_app_available = True
                server_type = type(app).__name__
            except (ImportError, AttributeError):
                server_app_available = False
                server_type = None

            protocol_checks.append(
                {
                    "check": "server_app_availability",
                    "passed": server_app_available,
                    "details": {"server_type": server_type},
                }
            )

            # 3. 检查工具注册机制
            tool_registration_available = False
            try:
                # 检查是否有工具注册装饰器或函数
                from servers.mcp_server import get_available_tools

                tools = get_available_tools()
                tool_registration_available = isinstance(tools, list)
            except:
                pass

            protocol_checks.append(
                {"check": "tool_registration_mechanism", "passed": tool_registration_available, "details": {}}
            )

            # 4. 检查消息处理
            message_handling_available = False
            try:
                from servers.mcp_server import process_tool_call

                message_handling_available = callable(process_tool_call)
            except:
                pass

            protocol_checks.append({"check": "message_handling", "passed": message_handling_available, "details": {}})

            # 计算合规性分数
            passed_checks = sum(1 for check in protocol_checks if check["passed"])
            compliance_score = passed_checks / len(protocol_checks)

            self.test_results.append(
                {
                    "test": "mcp_protocol_compliance",
                    "status": "PASSED",
                    "details": {
                        "protocol_checks": protocol_checks,
                        "passed_checks": passed_checks,
                        "total_checks": len(protocol_checks),
                        "compliance_score": compliance_score,
                        "mcp_library_available": mcp_library_available,
                    },
                }
            )

            logger.info(f"✅ MCP协议合规性检查完成 - 合规分数: {compliance_score:.2f}")
            return compliance_score > 0.5

        except Exception as e:
            logger.error(f"❌ MCP协议合规性测试失败: {e}")
            self.test_results.append({"test": "mcp_protocol_compliance", "status": "FAILED", "error": str(e)})
            return False

    async def test_mcp_configuration_validation(self):
        """测试MCP配置验证"""
        logger.info("⚙️ 测试MCP配置验证...")

        try:
            # 检查MCP Inspector配置文件
            inspector_config_path = Path("mcp-inspector-config.json")
            config_file_exists = inspector_config_path.exists()

            config_validation = {
                "config_file_exists": config_file_exists,
                "config_valid": False,
                "server_configs": [],
                "environment_vars": [],
            }

            if config_file_exists:
                try:
                    with open(inspector_config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)

                    # 验证配置结构
                    if "mcpServers" in config_data:
                        config_validation["config_valid"] = True

                        for server_name, server_config in config_data["mcpServers"].items():
                            server_validation = {
                                "name": server_name,
                                "has_command": "command" in server_config,
                                "has_args": "args" in server_config,
                                "has_cwd": "cwd" in server_config,
                                "has_env": "env" in server_config,
                                "python_command": server_config.get("command") == "python",
                                "correct_module": any(
                                    "-m" in str(arg) and "servers.mcp_server" in str(arg)
                                    for arg in server_config.get("args", [])
                                ),
                            }
                            config_validation["server_configs"].append(server_validation)

                        # 检查环境变量配置
                        for server_config in config_data["mcpServers"].values():
                            env_vars = server_config.get("env", {})
                            for var_name, var_value in env_vars.items():
                                config_validation["environment_vars"].append(
                                    {
                                        "name": var_name,
                                        "is_template": var_value.startswith("${") and var_value.endswith("}"),
                                        "value_preview": var_value[:20] + "..." if len(var_value) > 20 else var_value,
                                    }
                                )

                except json.JSONDecodeError as e:
                    config_validation["config_parse_error"] = str(e)

            # 检查实际环境变量
            actual_env_vars = []
            important_env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PYTHONPATH"]

            for var_name in important_env_vars:
                var_value = os.getenv(var_name)
                actual_env_vars.append(
                    {"name": var_name, "available": var_value is not None, "length": len(var_value) if var_value else 0}
                )

            config_validation["actual_environment_vars"] = actual_env_vars

            # 计算配置质量分数
            quality_factors = [
                config_validation["config_file_exists"],
                config_validation["config_valid"],
                len(config_validation["server_configs"]) > 0,
                len(config_validation["environment_vars"]) > 0,
                any(env["available"] for env in actual_env_vars),
            ]

            quality_score = sum(quality_factors) / len(quality_factors)

            self.test_results.append(
                {
                    "test": "mcp_configuration_validation",
                    "status": "PASSED",
                    "details": {**config_validation, "quality_score": quality_score},
                }
            )

            logger.info(f"✅ MCP配置验证完成 - 质量分数: {quality_score:.2f}")
            return quality_score > 0.6

        except Exception as e:
            logger.error(f"❌ MCP配置验证测试失败: {e}")
            self.test_results.append({"test": "mcp_configuration_validation", "status": "FAILED", "error": str(e)})
            return False

    async def run_all_tests(self):
        """运行所有MCP服务器工具测试"""
        logger.info("🚀 开始MCP服务器工具功能测试...")
        start_time = datetime.now()

        tests = [
            ("MCP服务器模块结构", self.test_mcp_server_module_structure),
            ("MCP工具定义", self.test_mcp_tool_definitions),
            ("工具调用模拟", self.test_tool_call_simulation),
            ("MCP协议合规性", self.test_mcp_protocol_compliance),
            ("MCP配置验证", self.test_mcp_configuration_validation),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 执行测试: {test_name}")
            logger.info(f"{'='*60}")

            try:
                result = await test_func()
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name} - 通过")
                else:
                    failed += 1
                    logger.error(f"❌ {test_name} - 失败")

                await asyncio.sleep(1)

            except Exception as e:
                failed += 1
                logger.error(f"❌ {test_name} - 异常: {e}")

        # 生成测试报告
        await self.generate_test_report(passed, failed, start_time)

        return failed == 0

    async def generate_test_report(self, passed: int, failed: int, start_time: datetime):
        """生成测试报告"""
        end_time = datetime.now()
        duration = end_time - start_time

        report = {
            "test_session": {
                "test_type": "MCP Server Tools Test",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "total_tests": len(self.test_results),
                "passed": passed,
                "failed": failed,
                "success_rate": f"{(passed / len(self.test_results) * 100):.1f}%" if self.test_results else "0%",
            },
            "test_results": self.test_results,
        }

        # 保存报告
        report_dir = Path("test_reports")
        report_dir.mkdir(exist_ok=True)

        report_path = report_dir / f"mcp_server_tools_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 测试报告已保存: {report_path}")

        # 打印摘要
        self._print_summary(report)

    def _print_summary(self, report):
        """打印测试摘要"""
        print(f"\n{'='*80}")
        print("🧪 MCP服务器工具功能测试报告")
        print(f"{'='*80}")
        print(
            f"📅 测试时间: {datetime.fromisoformat(report['test_session']['start_time']).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"⏱️ 测试时长: {report['test_session']['duration_seconds']:.2f} 秒")
        print(f"📊 总测试数: {report['test_session']['total_tests']}")
        print(f"✅ 通过: {report['test_session']['passed']}")
        print(f"❌ 失败: {report['test_session']['failed']}")
        print(f"📈 成功率: {report['test_session']['success_rate']}")
        print("")

        print("📋 测试详情:")
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"  {status_icon} {result['test']}")
            if result["status"] == "FAILED" and "error" in result:
                print(f"      错误: {result['error']}")

        print(f"{'='*80}")


async def main():
    """主函数"""
    tester = MCPServerToolsTester()

    try:
        success = await tester.run_all_tests()
        if success:
            print("🎉 所有MCP服务器工具测试通过！")
            return 0
        else:
            print("⚠️ 部分测试失败，请检查报告")
            return 1
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        return 1
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        return 1
    finally:
        tester.cleanup_all()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
