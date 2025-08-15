#!/usr/bin/env python3
"""
MCP Academic RAG Server - 完整集成测试
确保彻底清理资源，避免进程遗留
"""

import sys
import os
import json
import time
import asyncio
import logging
import signal
import atexit
import subprocess
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath('.'))

class TestResourceManager:
    """测试资源管理器 - 确保彻底清理"""
    
    def __init__(self):
        self.cleanup_functions = []
        self.processes = []
        self.files_to_cleanup = []
        self.setup_signal_handlers()
        atexit.register(self.cleanup_all)
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            print(f"\n收到信号 {signum}，开始清理...")
            self.cleanup_all()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def register_cleanup(self, func):
        """注册清理函数"""
        self.cleanup_functions.append(func)
    
    def register_process(self, proc):
        """注册进程用于清理"""
        self.processes.append(proc)
    
    def register_file(self, filepath):
        """注册临时文件用于清理"""
        self.files_to_cleanup.append(filepath)
    
    def cleanup_all(self):
        """执行所有清理"""
        print("🧹 开始资源清理...")
        
        # 清理注册的函数
        for func in self.cleanup_functions:
            try:
                if asyncio.iscoroutinefunction(func):
                    asyncio.create_task(func())
                else:
                    func()
            except Exception as e:
                print(f"清理函数失败: {e}")
        
        # 清理进程
        for proc in self.processes:
            try:
                if proc.poll() is None:  # 进程仍在运行
                    proc.terminate()
                    proc.wait(timeout=5)
            except Exception as e:
                try:
                    proc.kill()
                except:
                    pass
        
        # 清理文件
        for filepath in self.files_to_cleanup:
            try:
                Path(filepath).unlink(missing_ok=True)
            except Exception:
                pass
        
        # 额外的进程清理
        self._cleanup_remaining_processes()
        
        print("✅ 资源清理完成")
    
    def _cleanup_remaining_processes(self):
        """清理残留进程"""
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 'WINDOWTITLE eq *mcp*'], 
                              check=False, capture_output=True)
            else:  # Unix-like
                subprocess.run(['pkill', '-f', 'python.*mcp'], check=False)
                subprocess.run(['pkill', '-f', 'python.*test'], check=False)
        except Exception:
            pass

# 全局资源管理器
resource_manager = TestResourceManager()

# 配置日志
log_file = Path("test_reports/complete_integration_test.log")
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w')
    ]
)
logger = logging.getLogger(__name__)

class CompleteIntegrationTester:
    """完整集成测试器"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.test_results = []
        self.api_calls = 0
        self.errors = []
    
    async def test_1_environment_validation(self):
        """测试1: 环境验证"""
        logger.info("🧪 测试1: 环境验证")
        
        try:
            # 检查Python版本
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
                raise Exception(f"Python版本过低: {python_version}")
            
            # 检查API密钥
            api_keys = {
                'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
                'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY')
            }
            
            available_keys = [k for k, v in api_keys.items() if v]
            if not available_keys:
                raise Exception("没有可用的API密钥")
            
            logger.info(f"✅ Python版本: {sys.version}")
            logger.info(f"✅ 可用API密钥: {available_keys}")
            
            self.test_results.append({
                'test': 'environment_validation',
                'status': 'PASSED',
                'details': {
                    'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                    'available_apis': available_keys
                }
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 环境验证失败: {e}")
            self.test_results.append({
                'test': 'environment_validation',
                'status': 'FAILED',
                'error': str(e)
            })
            return False
    
    async def test_2_core_imports(self):
        """测试2: 核心模块导入"""
        logger.info("🧪 测试2: 核心模块导入")
        
        import_tests = [
            ('core.config_center', 'ConfigCenter'),
            ('document_stores', 'VectorStoreFactory'),
            ('connectors.base_llm_connector', 'BaseLLMConnector'),
            ('rag.haystack_pipeline', None),
            ('servers.mcp_server', None)
        ]
        
        imported_modules = []
        failed_imports = []
        
        for module_name, class_name in import_tests:
            try:
                module = __import__(module_name, fromlist=[class_name] if class_name else [])
                if class_name:
                    getattr(module, class_name)
                imported_modules.append(module_name)
                logger.info(f"✅ 导入成功: {module_name}")
            except Exception as e:
                failed_imports.append((module_name, str(e)))
                logger.error(f"❌ 导入失败: {module_name} - {e}")
        
        success = len(failed_imports) == 0
        
        self.test_results.append({
            'test': 'core_imports',
            'status': 'PASSED' if success else 'FAILED',
            'details': {
                'imported_modules': imported_modules,
                'failed_imports': failed_imports
            }
        })
        
        return success
    
    async def test_3_configuration_system(self):
        """测试3: 配置系统"""
        logger.info("🧪 测试3: 配置系统")
        
        try:
            from core.config_center import ConfigCenter
            
            # 创建配置中心实例
            config_center = ConfigCenter(
                base_config_path="./config",
                environment="test"
            )
            
            # 注册清理
            resource_manager.register_cleanup(
                lambda: config_center.cleanup() if hasattr(config_center, 'cleanup') else None
            )
            
            # 测试配置加载
            config = config_center.get_config()
            
            # 验证配置结构
            required_sections = ['server', 'llm', 'vector_db', 'processing']
            missing_sections = [s for s in required_sections if s not in config]
            
            if missing_sections:
                raise Exception(f"配置缺少必需部分: {missing_sections}")
            
            # 测试配置热更新
            original_temp = config['llm']['parameters']['temperature']
            config_center.set_value('llm.parameters.temperature', 0.5)
            updated_temp = config_center.get_value('llm.parameters.temperature')
            
            if updated_temp != 0.5:
                raise Exception("配置热更新失败")
            
            # 恢复原值
            config_center.set_value('llm.parameters.temperature', original_temp)
            
            logger.info(f"✅ 配置系统测试通过")
            logger.info(f"   LLM提供商: {config['llm']['provider']}")
            logger.info(f"   向量存储: {config['vector_db']['type']}")
            
            self.test_results.append({
                'test': 'configuration_system',
                'status': 'PASSED',
                'details': {
                    'llm_provider': config['llm']['provider'],
                    'vector_store_type': config['vector_db']['type'],
                    'hot_reload_test': 'passed'
                }
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置系统测试失败: {e}")
            self.test_results.append({
                'test': 'configuration_system',
                'status': 'FAILED',
                'error': str(e)
            })
            return False
    
    async def test_4_vector_storage(self):
        """测试4: 向量存储系统"""
        logger.info("🧪 测试4: 向量存储系统")
        
        try:
            from document_stores import VectorStoreFactory
            
            # 测试内存向量存储
            vector_config = {
                'type': 'memory',
                'vector_dimension': 384,
                'similarity': 'dot_product'
            }
            
            vector_store = VectorStoreFactory.create_store(vector_config)
            
            # 注册清理
            resource_manager.register_cleanup(
                lambda: vector_store.cleanup() if hasattr(vector_store, 'cleanup') else None
            )
            
            # 验证存储创建成功
            if not vector_store:
                raise Exception("向量存储创建失败")
            
            logger.info(f"✅ 向量存储创建成功: {type(vector_store).__name__}")
            
            self.test_results.append({
                'test': 'vector_storage',
                'status': 'PASSED',
                'details': {
                    'store_type': type(vector_store).__name__,
                    'config': vector_config
                }
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 向量存储测试失败: {e}")
            self.test_results.append({
                'test': 'vector_storage',
                'status': 'FAILED',
                'error': str(e)
            })
            return False
    
    async def test_5_document_processing(self):
        """测试5: 文档处理"""
        logger.info("🧪 测试5: 文档处理")
        
        try:
            # 验证测试文档
            test_docs = [
                "test-documents/research-paper-sample.txt",
                "test-documents/machine-learning.txt"
            ]
            
            processed_docs = []
            
            for doc_path in test_docs:
                doc_file = Path(doc_path)
                if not doc_file.exists():
                    logger.warning(f"⚠️ 测试文档不存在: {doc_path}")
                    continue
                
                content = doc_file.read_text(encoding='utf-8')
                if len(content) < 50:
                    logger.warning(f"⚠️ 文档内容过短: {doc_path}")
                    continue
                
                processed_docs.append({
                    'path': doc_path,
                    'size': len(content),
                    'lines': len(content.split('\n'))
                })
                
                logger.info(f"✅ 文档验证通过: {doc_path} ({len(content)} 字符)")
            
            if not processed_docs:
                raise Exception("没有有效的测试文档")
            
            self.test_results.append({
                'test': 'document_processing',
                'status': 'PASSED',
                'details': {
                    'processed_documents': processed_docs,
                    'total_documents': len(processed_docs)
                }
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 文档处理测试失败: {e}")
            self.test_results.append({
                'test': 'document_processing',
                'status': 'FAILED',
                'error': str(e)
            })
            return False
    
    async def test_6_api_connectivity(self):
        """测试6: API连接性 (真实调用)"""
        logger.info("🧪 测试6: API连接性 (真实调用)")
        
        api_results = {}
        
        # 测试OpenAI API
        if os.getenv('OPENAI_API_KEY'):
            try:
                result = await self._test_openai_real_call()
                api_results['openai'] = result
                self.api_calls += 1
            except Exception as e:
                api_results['openai'] = {'status': 'failed', 'error': str(e)}
        
        # 测试Anthropic API
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                result = await self._test_anthropic_real_call()
                api_results['anthropic'] = result
                self.api_calls += 1
            except Exception as e:
                api_results['anthropic'] = {'status': 'failed', 'error': str(e)}
        
        success = any(r.get('status') == 'success' for r in api_results.values())
        
        self.test_results.append({
            'test': 'api_connectivity',
            'status': 'PASSED' if success else 'FAILED',
            'details': {
                'api_results': api_results,
                'total_calls': self.api_calls
            }
        })
        
        return success
    
    async def _test_openai_real_call(self):
        """真实OpenAI API调用"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # 注册清理
            resource_manager.register_cleanup(lambda: client.close() if hasattr(client, 'close') else None)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Respond with exactly: API test successful"}
                ],
                max_tokens=10
            )
            
            result_text = response.choices[0].message.content
            logger.info(f"✅ OpenAI API响应: {result_text}")
            
            return {
                'status': 'success',
                'response': result_text,
                'model': 'gpt-3.5-turbo',
                'tokens_used': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI API调用失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _test_anthropic_real_call(self):
        """真实Anthropic API调用"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
            # 注册清理
            resource_manager.register_cleanup(lambda: client.close() if hasattr(client, 'close') else None)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[
                    {"role": "user", "content": "Respond with exactly: API test successful"}
                ]
            )
            
            result_text = response.content[0].text
            logger.info(f"✅ Anthropic API响应: {result_text}")
            
            return {
                'status': 'success', 
                'response': result_text,
                'model': 'claude-3-haiku-20240307',
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens
            }
            
        except Exception as e:
            logger.error(f"❌ Anthropic API调用失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def test_7_mcp_server_functionality(self):
        """测试7: MCP服务器功能"""
        logger.info("🧪 测试7: MCP服务器功能")
        
        try:
            # 导入MCP服务器
            from servers import mcp_server
            
            # 验证服务器应用存在
            if not hasattr(mcp_server, 'app'):
                raise Exception("MCP服务器应用未找到")
            
            # 这里可以添加更多MCP服务器功能测试
            logger.info("✅ MCP服务器模块导入成功")
            
            self.test_results.append({
                'test': 'mcp_server_functionality',
                'status': 'PASSED',
                'details': {
                    'server_module': 'servers.mcp_server',
                    'app_available': True
                }
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ MCP服务器功能测试失败: {e}")
            self.test_results.append({
                'test': 'mcp_server_functionality',
                'status': 'FAILED',
                'error': str(e)
            })
            return False
    
    async def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始完整集成测试...")
        
        tests = [
            ('环境验证', self.test_1_environment_validation),
            ('核心模块导入', self.test_2_core_imports),
            ('配置系统', self.test_3_configuration_system),
            ('向量存储系统', self.test_4_vector_storage),
            ('文档处理', self.test_5_document_processing),
            ('API连接性', self.test_6_api_connectivity),
            ('MCP服务器功能', self.test_7_mcp_server_functionality)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"执行测试: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = await test_func()
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name} - 通过")
                else:
                    failed += 1
                    logger.error(f"❌ {test_name} - 失败")
                
                # 测试间延迟
                await asyncio.sleep(1)
                
            except Exception as e:
                failed += 1
                logger.error(f"❌ {test_name} - 异常: {e}")
                self.errors.append(f"{test_name}: {e}")
        
        # 生成报告
        await self.generate_test_report(passed, failed)
        
        return failed == 0
    
    async def generate_test_report(self, passed: int, failed: int):
        """生成测试报告"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            'test_session': {
                'test_type': 'Complete Integration Test',
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'total_tests': len(self.test_results),
                'passed': passed,
                'failed': failed,
                'success_rate': f"{(passed / len(self.test_results) * 100):.1f}%" if self.test_results else "0%",
                'api_calls_made': self.api_calls
            },
            'environment': {
                'python_version': sys.version,
                'platform': os.name,
                'working_directory': os.getcwd()
            },
            'test_results': self.test_results,
            'errors': self.errors
        }
        
        # 保存报告
        report_dir = Path("test_reports/regression")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"complete_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 测试报告已保存: {report_path}")
        
        # 打印控制台摘要
        self._print_summary(report)
    
    def _print_summary(self, report):
        """打印测试摘要"""
        print("\n" + "="*80)
        print("🧪 MCP Academic RAG Server - 完整集成测试报告")
        print("="*80)
        print(f"📅 测试时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ 测试时长: {report['test_session']['duration_seconds']:.2f} 秒")
        print(f"📊 总测试数: {report['test_session']['total_tests']}")
        print(f"✅ 通过: {report['test_session']['passed']}")
        print(f"❌ 失败: {report['test_session']['failed']}")
        print(f"📈 成功率: {report['test_session']['success_rate']}")
        print(f"🔑 API调用: {report['test_session']['api_calls_made']} 次")
        print("")
        
        print("📋 测试详情:")
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"  {status_icon} {result['test']}")
            if result['status'] == 'FAILED' and 'error' in result:
                print(f"      错误: {result['error']}")
        
        print("="*80)

async def main():
    """主函数"""
    print("🧪 MCP Academic RAG Server - 完整集成测试")
    print("✨ 包含真实API调用和完整资源清理")
    print("")
    
    tester = CompleteIntegrationTester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            print("🎉 所有测试通过！")
            return 0
        else:
            print("⚠️ 部分测试失败，请检查报告")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        return 1
    except Exception as e:
        print(f"\n❌ 测试执行异常: {e}")
        return 1
    finally:
        # 确保资源清理
        resource_manager.cleanup_all()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)