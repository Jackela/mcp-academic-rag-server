#!/usr/bin/env python3
"""
MCP Academic RAG Server - 快速真实API测试
包含资源清理，避免进程遗留
"""

import asyncio
import atexit
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, os.path.abspath("."))

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QuickAPITester:
    """快速API测试器"""

    def __init__(self):
        self.cleanup_functions = []
        self.setup_signal_handlers()

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
                    # 对于异步函数，简单调用（不完美但避免复杂性）
                    pass
                else:
                    func()
            except Exception as e:
                logger.error(f"清理函数失败: {e}")
        logger.info("✅ 资源清理完成")

    async def test_openai_api(self):
        """测试OpenAI API"""
        logger.info("🤖 测试OpenAI API...")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("❌ OPENAI_API_KEY 未设置")
            return False

        try:
            import openai

            client = openai.OpenAI(api_key=api_key)

            # 注册清理
            self.register_cleanup(lambda: client.close() if hasattr(client, "close") else None)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Respond with exactly: API test successful"}],
                max_tokens=10,
            )

            result = response.choices[0].message.content
            logger.info(f"✅ OpenAI API响应: {result}")
            return True

        except Exception as e:
            logger.error(f"❌ OpenAI API测试失败: {e}")
            return False

    async def test_document_processing(self):
        """测试文档处理"""
        logger.info("📄 测试文档处理...")

        doc_path = Path("test-documents/research-paper-sample.txt")
        if not doc_path.exists():
            logger.error("❌ 测试文档不存在")
            return False

        try:
            content = doc_path.read_text(encoding="utf-8")
            if len(content) < 100:
                logger.error("❌ 文档内容过短")
                return False

            logger.info(f"✅ 文档读取成功: {len(content)} 字符")
            return True

        except Exception as e:
            logger.error(f"❌ 文档处理失败: {e}")
            return False

    async def test_config_loading(self):
        """测试配置加载"""
        logger.info("⚙️ 测试配置加载...")

        try:
            from core.config_center import ConfigCenter

            config_center = ConfigCenter(base_config_path="./config", environment="test")

            # 注册清理
            self.register_cleanup(lambda: config_center.cleanup() if hasattr(config_center, "cleanup") else None)

            config = config_center.get_config()

            if "llm" not in config or "vector_db" not in config:
                logger.error("❌ 配置结构不完整")
                return False

            logger.info(f"✅ 配置加载成功: LLM={config['llm']['provider']}")
            return True

        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            return False

    async def run_quick_tests(self):
        """运行快速测试"""
        logger.info("🚀 开始快速真实API测试...")

        tests = [
            ("配置加载", self.test_config_loading),
            ("文档处理", self.test_document_processing),
            ("OpenAI API", self.test_openai_api),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 执行测试: {test_name}")
            logger.info(f"{'='*50}")

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

        # 打印结果
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0

        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 快速测试完成")
        logger.info(f"📊 总测试数: {total}")
        logger.info(f"✅ 通过: {passed}")
        logger.info(f"❌ 失败: {failed}")
        logger.info(f"📈 成功率: {success_rate:.1f}%")
        logger.info(f"{'='*60}")

        return failed == 0


async def main():
    """主函数"""
    tester = QuickAPITester()

    try:
        success = await tester.run_quick_tests()
        if success:
            print("🎉 所有快速测试通过！")
            return 0
        else:
            print("⚠️ 部分测试失败")
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
