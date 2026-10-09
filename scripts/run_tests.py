#!/usr/bin/env python3
"""
向量存储测试运行脚本

提供不同类型测试的便捷运行方式，支持测试筛选、报告生成和结果分析。
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


class TestRunner:
    """测试运行器类"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results_dir = self.project_root / "test-results"
        self.coverage_dir = self.project_root / "htmlcov"

        # 确保结果目录存在
        self.test_results_dir.mkdir(exist_ok=True)

    def run_command(self, command: List[str], description: str = None) -> bool:
        """运行命令并返回是否成功"""
        if description:
            print(f"\n🔧 {description}")

        print(f"Running: {' '.join(command)}")

        try:
            result = subprocess.run(command, cwd=self.project_root, capture_output=False, text=True)

            success = result.returncode == 0
            if success:
                print(f"✅ {description or 'Command'} completed successfully")
            else:
                print(f"❌ {description or 'Command'} failed with code {result.returncode}")

            return success

        except Exception as e:
            print(f"❌ Error running {description or 'command'}: {e}")
            return False

    def check_dependencies(self) -> Dict[str, bool]:
        """检查测试依赖是否可用"""
        dependencies = {}

        # 检查基础依赖
        try:
            __import__("pytest")

            dependencies["pytest"] = True
        except ImportError:
            dependencies["pytest"] = False

        # 检查FAISS
        try:
            __import__("faiss")

            dependencies["faiss"] = True
        except ImportError:
            dependencies["faiss"] = False

        # 检查Milvus客户端
        try:
            __import__("pymilvus")

            dependencies["pymilvus"] = True
        except ImportError:
            dependencies["pymilvus"] = False

        # 检查覆盖率工具
        try:
            __import__("coverage")

            dependencies["coverage"] = True
        except ImportError:
            dependencies["coverage"] = False

        return dependencies

    def print_dependency_status(self):
        """打印依赖状态"""
        print("\n📋 Dependency Status:")
        dependencies = self.check_dependencies()

        for dep, available in dependencies.items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep}")

        if not dependencies.get("pytest", False):
            print("\n❌ PyTest is required. Install with: pip install pytest")
            return False

        return True

    def run_unit_tests(self, verbose: bool = True, coverage: bool = True) -> bool:
        """运行单元测试"""
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10",
            f"--junitxml={self.test_results_dir / 'junit-unit.xml'}",
            "-m",
            "not slow",
        ]

        if coverage:
            command.extend(
                [
                    "--cov=document_stores",
                    "--cov=utils.vector_migration",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                    f"--cov-report=xml:{self.test_results_dir / 'coverage-unit.xml'}",
                ]
            )

        return self.run_command(command, "Running unit tests")

    def run_integration_tests(self, verbose: bool = True, coverage: bool = True) -> bool:
        """运行集成测试"""
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10",
            f"--junitxml={self.test_results_dir / 'junit-integration.xml'}",
            "-m",
            "not slow and not requires_milvus",
        ]

        if coverage:
            command.extend(
                [
                    "--cov=document_stores",
                    "--cov=utils.vector_migration",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                    f"--cov-report=xml:{self.test_results_dir / 'coverage-integration.xml'}",
                ]
            )

        return self.run_command(command, "Running integration tests")

    def run_performance_tests(self, verbose: bool = True) -> bool:
        """运行性能测试"""
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/performance/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=20",
            f"--junitxml={self.test_results_dir / 'junit-performance.xml'}",
            "-m",
            "benchmark",
            "--timeout=300",
        ]

        return self.run_command(command, "Running performance tests")

    def run_all_tests(self, verbose: bool = True, coverage: bool = True) -> bool:
        """运行所有测试"""
        print("\n🚀 Running all vector storage tests...\n")

        success = True

        # 单元测试
        if not self.run_unit_tests(verbose, coverage):
            success = False

        # 集成测试
        if not self.run_integration_tests(verbose, coverage):
            success = False

        return success

    def run_quick_tests(self) -> bool:
        """运行快速测试（跳过慢速测试）"""
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/",
            "tests/integration/",
            "-q",
            "--tb=line",
            "-m",
            "not slow and not benchmark and not requires_milvus",
            "--maxfail=5",
        ]

        return self.run_command(command, "Running quick tests")

    def run_specific_tests(self, pattern: str, verbose: bool = True) -> bool:
        """运行特定测试"""
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-k",
            pattern,
            "-v" if verbose else "-q",
            "--tb=short",
            f"--junitxml={self.test_results_dir / 'junit-specific.xml'}",
        ]

        return self.run_command(command, f"Running tests matching '{pattern}'")

    def run_tests_with_markers(self, markers: List[str], verbose: bool = True) -> bool:
        """根据标记运行测试"""
        marker_expr = " and ".join(markers)

        command = [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            marker_expr,
            "-v" if verbose else "-q",
            "--tb=short",
            f"--junitxml={self.test_results_dir / 'junit-markers.xml'}",
        ]

        return self.run_command(command, f"Running tests with markers: {marker_expr}")

    def generate_coverage_report(self) -> bool:
        """生成覆盖率报告"""
        if not self.coverage_dir.exists():
            print("❌ No coverage data found. Run tests with coverage first.")
            return False

        print(f"\n📊 Coverage report available at: {self.coverage_dir / 'index.html'}")

        # 尝试打开覆盖率报告
        try:
            import webbrowser

            webbrowser.open(f"file://{self.coverage_dir / 'index.html'}")
            print("🌐 Coverage report opened in browser")
        except Exception:
            print("📂 Open the file manually to view the report")

        return True

    def clean_test_artifacts(self) -> bool:
        """清理测试产生的文件"""
        import shutil

        artifacts = [
            self.test_results_dir,
            self.coverage_dir,
            self.project_root / ".coverage",
            self.project_root / ".pytest_cache",
            self.project_root / "coverage.xml",
        ]

        removed = 0
        for artifact in artifacts:
            if artifact.exists():
                try:
                    if artifact.is_dir():
                        shutil.rmtree(artifact)
                    else:
                        artifact.unlink()
                    removed += 1
                    print(f"🗑️ Removed {artifact}")
                except Exception as e:
                    print(f"⚠️ Could not remove {artifact}: {e}")

        if removed > 0:
            print(f"\n✅ Cleaned {removed} test artifacts")
        else:
            print("✅ No test artifacts to clean")

        return True

    def print_test_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("🧪 Vector Storage Test Suite")
        print("=" * 60)
        print(f"📁 Project root: {self.project_root}")
        print(f"📊 Results dir: {self.test_results_dir}")
        print(f"📈 Coverage dir: {self.coverage_dir}")
        print("=" * 60)


def _run_selected_tests(args, runner, verbose, coverage):
    # 运行测试
    if args.all:
        return runner.run_all_tests(verbose, coverage)
    elif args.unit:
        return runner.run_unit_tests(verbose, coverage)
    elif args.integration:
        return runner.run_integration_tests(verbose, coverage)
    elif args.performance:
        return runner.run_performance_tests(verbose)
    elif args.quick:
        return runner.run_quick_tests()
    elif args.pattern:
        return runner.run_specific_tests(args.pattern, verbose)
    elif args.markers:
        return runner.run_tests_with_markers(args.markers, verbose)
    else:
        # 默认运行快速测试
        print("No specific test type specified, running quick tests...")
        return runner.run_quick_tests()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Vector Storage Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py --all                    # 运行所有测试
  python scripts/run_tests.py --unit                   # 仅运行单元测试
  python scripts/run_tests.py --integration            # 仅运行集成测试
  python scripts/run_tests.py --performance            # 仅运行性能测试
  python scripts/run_tests.py --quick                  # 运行快速测试
  python scripts/run_tests.py --pattern "faiss"        # 运行包含"faiss"的测试
  python scripts/run_tests.py --markers "unit"         # 运行标记为unit的测试
  python scripts/run_tests.py --clean                  # 清理测试产生的文件
  python scripts/run_tests.py --deps                   # 检查依赖状态
        """,
    )

    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--unit", action="store_true", help="运行单元测试")
    parser.add_argument("--integration", action="store_true", help="运行集成测试")
    parser.add_argument("--performance", action="store_true", help="运行性能测试")
    parser.add_argument("--quick", action="store_true", help="运行快速测试")

    parser.add_argument("--pattern", type=str, help="运行匹配模式的测试")
    parser.add_argument("--markers", type=str, nargs="+", help="根据标记运行测试")

    parser.add_argument("--no-coverage", action="store_true", help="不生成覆盖率报告")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    parser.add_argument("--coverage-report", action="store_true", help="生成覆盖率报告")
    parser.add_argument("--clean", action="store_true", help="清理测试产生的文件")
    parser.add_argument("--deps", action="store_true", help="检查测试依赖")

    args = parser.parse_args()

    runner = TestRunner()
    runner.print_test_summary()

    # 处理特殊命令
    if args.deps:
        runner.print_dependency_status()
        return

    if args.clean:
        runner.clean_test_artifacts()
        return

    if args.coverage_report:
        runner.generate_coverage_report()
        return

    # 检查依赖
    if not runner.print_dependency_status():
        sys.exit(1)

    # 设置参数
    verbose = not args.quiet
    if args.verbose:
        verbose = True

    coverage = not args.no_coverage
    success = _run_selected_tests(args, runner, verbose, coverage)

    # 打印结果
    if success:
        print("\n🎉 All tests completed successfully!")
        if coverage and not args.no_coverage:
            runner.generate_coverage_report()
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
