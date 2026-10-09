#!/usr/bin/env python3
"""
CI/CD失败检测测试

这个测试文件包含一个故意失败的测试用例，用于验证CI/CD工作流程
能够正确识别和报告测试失败。

运行时通过环境变量控制是否启用失败测试。
"""

import os
import sys
import unittest

import pytest


class TestCIFailureDetection(unittest.TestCase):
    """CI/CD失败检测测试类"""

    def test_normal_passing_test(self):
        """这是一个正常通过的测试"""
        self.assertTrue(True, "这个测试应该总是通过")
        self.assertEqual(2 + 2, 4, "基本数学运算应该正确")

    def test_conditional_failure(self):
        """条件性失败测试 - 根据环境变量决定是否失败"""
        # 检查是否启用了失败测试
        enable_failure = os.environ.get("ENABLE_FAILURE_TEST", "false").lower()

        if enable_failure in ["true", "1", "yes", "on"]:
            # 故意失败
            self.fail("这是一个故意失败的测试，用于验证CI/CD工作流程的失败检测功能")
        else:
            # 正常通过
            self.assertTrue(True, "失败测试未启用，测试通过")

    def test_assertion_error_detection(self):
        """断言错误检测测试"""
        enable_assertion_error = os.environ.get("ENABLE_ASSERTION_ERROR", "false").lower()

        if enable_assertion_error in ["true", "1", "yes", "on"]:
            # 触发断言错误
            self.assertEqual(1, 2, "1不等于2 - 这是一个故意的断言错误")
        else:
            self.assertEqual(1, 1, "1等于1 - 正常断言")

    def test_exception_detection(self):
        """异常检测测试"""
        enable_exception = os.environ.get("ENABLE_EXCEPTION_TEST", "false").lower()

        if enable_exception in ["true", "1", "yes", "on"]:
            # 触发异常
            raise ValueError("这是一个故意抛出的异常，用于测试异常检测")
        else:
            # 正常执行
            result = 10 / 2
            self.assertEqual(result, 5.0, "数学运算正常")

    def test_import_error_simulation(self):
        """模拟导入错误"""
        enable_import_error = os.environ.get("ENABLE_IMPORT_ERROR", "false").lower()

        if enable_import_error in ["true", "1", "yes", "on"]:
            # 尝试导入不存在的模块
            try:
                import nonexistent_module_for_testing

                self.fail("不应该能够导入不存在的模块")
            except ImportError:
                # 将ImportError转换为测试失败
                self.fail("模拟导入错误 - 这是故意的失败")
        else:
            # 导入存在的模块
            import sys

            self.assertIsNotNone(sys, "系统模块应该可以正常导入")

    def test_timeout_simulation(self):
        """模拟超时测试"""
        import time

        enable_timeout = os.environ.get("ENABLE_TIMEOUT_TEST", "false").lower()
        timeout_duration = int(os.environ.get("TIMEOUT_DURATION", "1"))

        if enable_timeout in ["true", "1", "yes", "on"]:
            # 模拟长时间运行的测试
            time.sleep(timeout_duration)
            # 如果测试没有被超时机制中断，则失败
            self.fail(f"测试应该在{timeout_duration}秒内超时")
        else:
            # 快速完成
            time.sleep(0.01)
            self.assertTrue(True, "快速测试完成")


class TestCIEnvironmentValidation(unittest.TestCase):
    """CI环境验证测试"""

    def test_ci_environment_detection(self):
        """检测CI环境"""
        ci_indicators = ["CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "JENKINS_URL", "TRAVIS", "CIRCLECI"]

        is_ci = any(os.environ.get(indicator) for indicator in ci_indicators)

        if is_ci:
            print("✅ 检测到CI环境")
            # 在CI环境中可以有更严格的测试
            self.assertIsNotNone(os.environ.get("GITHUB_ACTIONS"), "应在GitHub Actions中运行")
        else:
            print("ℹ️ 本地开发环境")
            # 本地环境可以有更宽松的测试
            self.assertTrue(True, "本地环境测试通过")

    def test_required_environment_variables(self):
        """测试必需的环境变量"""
        # 检查测试环境变量
        testing_env = os.environ.get("TESTING", "false").lower()
        if testing_env in ["true", "1", "yes", "on"]:
            print("✅ 测试环境已启用")
        else:
            print("⚠️ 测试环境未明确启用")

        # 这个测试总是通过，只是为了验证环境
        self.assertTrue(True, "环境变量检查完成")

    def test_python_version_compatibility(self):
        """测试Python版本兼容性"""
        import sys

        major, minor = sys.version_info[:2]

        # 检查Python版本
        if major == 3 and minor >= 8:
            print(f"✅ Python {major}.{minor} 版本兼容")
            self.assertTrue(True, f"Python {major}.{minor} 支持")
        else:
            self.fail(f"Python {major}.{minor} 版本不受支持（需要3.8+）")

    def test_dependencies_availability(self):
        """测试依赖项可用性"""
        required_modules = ["numpy", "pandas", "torch", "haystack"]

        missing_modules = []

        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ {module} 可用")
            except ImportError:
                missing_modules.append(module)
                print(f"❌ {module} 不可用")

        # 在CI环境中，所有依赖都应该可用
        if os.environ.get("CI"):
            self.assertEqual(len(missing_modules), 0, f"CI环境中缺少依赖: {missing_modules}")
        else:
            # 本地环境允许部分依赖缺失
            if missing_modules:
                print(f"⚠️ 本地环境缺少依赖: {missing_modules}")


@pytest.mark.skipif(
    os.environ.get("SKIP_FAILURE_TESTS", "true").lower() in ["true", "1", "yes", "on"],
    reason="失败测试默认跳过，通过环境变量启用",
)
class TestPytestFailureDetection:
    """使用pytest的失败检测测试"""

    def test_pytest_assertion_failure(self):
        """pytest断言失败测试"""
        enable_failure = os.environ.get("ENABLE_PYTEST_FAILURE", "false").lower()

        if enable_failure in ["true", "1", "yes", "on"]:
            assert 1 == 2, "这是一个故意的pytest断言失败"
        else:
            assert 1 == 1, "正常pytest断言"

    def test_pytest_exception(self):
        """pytest异常测试"""
        enable_exception = os.environ.get("ENABLE_PYTEST_EXCEPTION", "false").lower()

        if enable_exception in ["true", "1", "yes", "on"]:
            raise RuntimeError("这是一个故意的pytest异常")
        else:
            # 正常执行
            result = [1, 2, 3]
            assert len(result) == 3

    @pytest.mark.parametrize("value,expected", [(1, 1), (2, 2), (3, 3)])
    def test_parametrized_test(self, value, expected):
        """参数化测试"""
        enable_param_failure = os.environ.get("ENABLE_PARAM_FAILURE", "false").lower()

        if enable_param_failure in ["true", "1", "yes", "on"] and value == 2:
            # 让第二个参数故意失败
            assert value == 999, f"参数化测试故意失败: {value}"
        else:
            assert value == expected


def simulate_test_failure():
    """
    手动触发测试失败的函数

    这个函数可以被CI脚本调用来测试失败检测机制
    """
    print("🔥 模拟测试失败...")

    # 设置环境变量以启用失败测试
    os.environ["ENABLE_FAILURE_TEST"] = "true"
    os.environ["ENABLE_ASSERTION_ERROR"] = "true"
    os.environ["ENABLE_EXCEPTION_TEST"] = "true"

    # 运行测试
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"返回码: {result.returncode}")

    return result.returncode != 0  # 如果测试失败返回True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CI/CD失败检测测试")
    parser.add_argument("--simulate-failure", action="store_true", help="模拟测试失败")
    parser.add_argument("--enable-all-failures", action="store_true", help="启用所有失败测试")

    args = parser.parse_args()

    if args.simulate_failure:
        failed = simulate_test_failure()
        print(f"测试失败模拟: {'成功' if failed else '失败'}")
        sys.exit(0 if failed else 1)

    if args.enable_all_failures:
        # 启用所有失败测试
        os.environ["ENABLE_FAILURE_TEST"] = "true"
        os.environ["ENABLE_ASSERTION_ERROR"] = "true"
        os.environ["ENABLE_EXCEPTION_TEST"] = "true"
        os.environ["ENABLE_IMPORT_ERROR"] = "true"
        os.environ["ENABLE_PYTEST_FAILURE"] = "true"
        os.environ["ENABLE_PYTEST_EXCEPTION"] = "true"
        os.environ["ENABLE_PARAM_FAILURE"] = "true"
        os.environ["SKIP_FAILURE_TESTS"] = "false"
        print("🔥 已启用所有失败测试")

    # 运行unittest
    unittest.main(verbosity=2)
