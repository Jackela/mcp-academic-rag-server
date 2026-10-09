"""
Automated Testing Integration Tests

Test suite for validating the automated testing infrastructure and CI/CD integration.
Includes test discovery, execution, reporting, and integration with various testing frameworks.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.utils.cleanup import ResourceCleaner, managed_resource, register_process


def _run_command(args, capture_output=False, text=False, cwd=None, env=None, input=None):
    """Bound children and release only this registered child on every exit path."""
    process = subprocess.Popen(
        args,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
        cwd=cwd,
        env=env,
    )
    register_process(process)
    try:
        stdout, stderr = process.communicate(input=input, timeout=60)
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class TestDiscoveryAndExecution:
    """Test automated test discovery and execution"""

    def test_pytest_discovery(self):
        """Test that pytest can discover all test files correctly"""
        # Run pytest with collection-only to discover tests
        result = _run_command(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        assert result.returncode == 0
        output = result.stdout

        # Should discover tests from various categories
        assert "test_unit" in output or "unit" in output
        assert "test_integration" in output or "integration" in output
        assert "test_e2e" in output or "e2e" in output

    def test_test_categorization(self):
        """Test that tests are properly categorized with markers"""
        # Check that pytest markers are properly configured
        result = _run_command(
            [sys.executable, "-m", "pytest", "--markers"], capture_output=True, text=True, cwd=Path.cwd()
        )

        assert result.returncode == 0
        markers_output = result.stdout

        # Should have custom markers defined
        expected_markers = ["unit", "integration", "e2e", "performance", "slow"]
        for marker in expected_markers:
            # Either marker is defined or pytest shows the custom marker
            assert marker in markers_output or f"@pytest.mark.{marker}" in markers_output

    def test_parallel_test_execution(self):
        """The declared xdist dependency must run actual tests successfully."""
        result = _run_command(
            [sys.executable, "-m", "pytest", "tests/unit/test_config_manager.py", "-n", "2", "-v"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "2 workers" in result.stdout


class TestTestReporting:
    """Test automated test reporting functionality"""

    def test_junit_xml_generation(self):
        """Test JUnit XML report generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            junit_file = Path(temp_dir) / "junit.xml"

            # Run a simple test with JUnit XML output
            result = _run_command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/unit/test_config_manager.py",
                    f"--junit-xml={junit_file}",
                    "-v",
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            assert result.returncode == 0, result.stdout + result.stderr
            # Should generate JUnit XML file
            assert junit_file.exists()

            # Verify XML content
            xml_content = junit_file.read_text()
            assert "<?xml version=" in xml_content
            assert "<testsuite" in xml_content
            assert "</testsuite>" in xml_content

    def test_coverage_reporting(self):
        """A successful child run must produce parseable coverage evidence."""
        from xml.etree import ElementTree

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "coverage.xml"
            result = _run_command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/unit/test_config_manager.py",
                    "--cov=core",
                    f"--cov-report=xml:{target}",
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            assert result.returncode == 0, result.stdout + result.stderr
            root = ElementTree.parse(target).getroot()
            assert root.tag == "coverage"
            assert int(root.attrib["lines-valid"]) > 0

    def test_html_report_generation(self):
        """The declared HTML plugin must produce a successful report."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.html"
            result = _run_command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/unit/test_config_manager.py",
                    f"--html={target}",
                    "--self-contained-html",
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            assert result.returncode == 0, result.stdout + result.stderr
            html = target.read_text()
            assert "<html" in html
            assert "pytest" in html or "Test Report" in html


class TestCIIntegration:
    """Test CI/CD integration capabilities"""

    def test_github_actions_workflow_syntax(self):
        """Test GitHub Actions workflow file syntax"""
        workflow_file = Path(".github/workflows/vector-storage-tests.yml")

        if workflow_file.exists():
            import yaml

            try:
                with open(workflow_file) as f:
                    workflow_content = yaml.load(f, Loader=yaml.BaseLoader)

                # Verify basic workflow structure
                assert "name" in workflow_content
                assert "on" in workflow_content
                assert "jobs" in workflow_content

                # Check for essential elements
                jobs = workflow_content["jobs"]
                assert len(jobs) > 0

                # At least one job should have steps
                first_job = next(iter(jobs.values()))
                assert "steps" in first_job
                assert len(first_job["steps"]) > 0

            except yaml.YAMLError:
                pytest.fail("GitHub Actions workflow file has invalid YAML syntax")
        else:
            pytest.fail("Required GitHub Actions workflow file not found")

    def test_environment_variable_handling(self):
        """Test handling of environment variables in testing"""
        # Test that tests can handle different environments
        original_env = os.environ.get("TEST_ENV")

        try:
            # Set test environment
            os.environ["TEST_ENV"] = "testing"

            # Run tests that might use environment variables
            result = _run_command(
                [sys.executable, "-m", "pytest", "tests/unit/test_config_system.py", "-v", "-k", "test_environment"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            # Should handle environment variables properly
            assert result.returncode == 0, result.stdout + result.stderr

        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["TEST_ENV"] = original_env
            else:
                os.environ.pop("TEST_ENV", None)

    def test_test_isolation(self):
        """Test that tests are properly isolated"""
        # Run the same test multiple times to check for side effects
        test_commands = [
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_config_manager.py::TestConfigManager::test_load_config",
                "-v",
            ],
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_config_manager.py::TestConfigManager::test_load_config",
                "-v",
            ],
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_config_manager.py::TestConfigManager::test_load_config",
                "-v",
            ],
        ]

        results = []
        for cmd in test_commands:
            result = _run_command(cmd, capture_output=True, text=True, cwd=Path.cwd())
            results.append(result.returncode)

        # All runs should have the same result (proper isolation)
        assert results == [0, 0, 0], results


class TestPerformanceTesting:
    """Test performance testing capabilities"""

    @pytest.mark.performance
    def test_performance_test_execution(self):
        """Test execution of performance tests"""
        # Check if performance tests exist and can be run
        perf_test_file = Path("tests/performance")

        if perf_test_file.exists():
            result = _run_command(
                [sys.executable, "-m", "pytest", "tests/performance/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            # Performance tests should execute (may pass or fail)
            assert result.returncode == 0, result.stdout + result.stderr

            # Should have run some tests
            assert "test session starts" in result.stdout
        else:
            pytest.fail("Required performance tests not found")

    @pytest.mark.performance
    def test_benchmark_integration(self):
        """Exercise a real benchmark fixture rather than skip custom performance tests."""
        with tempfile.TemporaryDirectory() as directory:
            test_file = Path(directory) / "test_benchmark_fixture.py"
            test_file.write_text("def test_real_benchmark(benchmark):\n    assert benchmark(lambda: 1 + 1) == 2\n")
            target = Path(directory) / "benchmark.json"
            result = _run_command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_file),
                    "--benchmark-only",
                    "--benchmark-max-time=0.01",
                    f"--benchmark-json={target}",
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            assert result.returncode == 0, result.stdout + result.stderr
            report = json.loads(target.read_text())
            assert len(report["benchmarks"]) == 1
            assert report["benchmarks"][0]["stats"]["rounds"] >= 1


class TestResourceManagement:
    """Test resource management during testing"""

    def test_resource_cleanup_integration(self):
        """Test that resource cleanup works correctly in tests"""
        cleaner = ResourceCleaner()

        # Test that cleanup context manager works
        with managed_resource(cleaner, cleanup_func=lambda resource: resource.cleanup_sync()):
            # Simulate resource creation
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()

            # Add to cleanup
            cleaner.register_cleanup(lambda: os.unlink(temp_file.name))

            # Verify file exists
            assert os.path.exists(temp_file.name)

        assert not os.path.exists(temp_file.name)

    def test_memory_leak_detection(self):
        """Test memory leak detection in test suite"""
        try:
            import gc

            import psutil

            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss

            # Run a subset of tests
            result = _run_command(
                [sys.executable, "-m", "pytest", "tests/unit/test_config_manager.py", "-v"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            assert result.returncode == 0, result.stdout + result.stderr
            # Force garbage collection
            gc.collect()

            # Check final memory usage
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 50MB for unit tests)
            assert memory_increase < 50 * 1024 * 1024  # 50MB threshold

        except ImportError:
            pytest.fail("Declared psutil dependency is unavailable")

    def test_test_data_cleanup(self):
        """Test that test data is properly cleaned up"""
        # Create temporary test data directory
        test_data_dir = Path("test_temp_data")

        try:
            test_data_dir.mkdir(exist_ok=True)
            test_file = test_data_dir / "test_file.txt"
            test_file.write_text("test data")

            # Run tests that might use test data
            result = _run_command(
                [sys.executable, "-m", "pytest", "tests/unit/test_config_manager.py", "-v"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            # Tests should execute successfully
            assert result.returncode == 0, result.stdout + result.stderr

        finally:
            # Cleanup test data
            if test_data_dir.exists():
                import shutil

                shutil.rmtree(test_data_dir, ignore_errors=True)


class TestFailureAnalysis:
    """Test failure analysis and debugging capabilities"""

    def test_failure_report_generation(self):
        """Test generation of detailed failure reports"""
        # Create a test that will definitely fail
        failing_test_content = '''
import pytest

def test_intentional_failure():
    """This test is designed to fail for testing failure reporting"""
    assert False, "Intentional failure for testing"

def test_exception_failure():
    """This test raises an exception"""
    raise ValueError("Test exception for failure analysis")
'''

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_failing.py"
            test_file.write_text(failing_test_content)

            # Run the failing test with verbose output
            result = _run_command(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=long"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            # Should have failure information
            assert result.returncode == 1  # Tests failed
            assert "FAILED" in result.stdout
            assert "Intentional failure" in result.stdout
            assert "ValueError" in result.stdout

    def test_debug_mode_integration(self):
        """Test debug mode integration for test failures"""
        # Test that debug flags work properly
        result = _run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_config_manager.py",
                "--trace",
                "--capture=no",
                "-x",  # Stop on first failure
            ],
            capture_output=True,
            text=True,
            input="quit\n",
            cwd=Path.cwd(),
        )

        assert result.returncode == 2  # Explicit debugger quit, not a test success
        assert "(Pdb)" in result.stdout

    def test_test_result_analysis(self):
        """Test analysis of test results"""
        # Run tests and capture detailed output
        result = _run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_config_manager.py",
                "--tb=short",
                "-v",
                "--durations=10",  # Show slowest 10 tests
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        assert result.returncode == 0, result.stdout + result.stderr
        output = result.stdout

        # Should provide test duration information
        if "durations" in output.lower() or "slowest" in output.lower():
            # Duration reporting is working
            assert True
        else:
            # Basic test execution information should be available
            assert "test session starts" in output
            assert "collected" in output


class TestContinuousIntegration:
    """Test continuous integration workflow"""

    def test_pre_commit_hook_simulation(self):
        """Test simulation of pre-commit hooks"""
        # Simulate running tests as a pre-commit hook
        result = _run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_config_manager.py",
                "-o",
                "addopts=",
                "-o",
                "log_cli=false",
                "--quiet",
                "--tb=no",
                "--no-cov",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        # Pre-commit style should be fast and minimal output
        assert result.returncode == 0, result.stdout + result.stderr
        execution_time = len(result.stdout.split("\n"))
        assert execution_time < 50  # Should have minimal output lines

    def test_test_selection_strategies(self):
        """Test different test selection strategies"""
        strategies = [
            # Run only unit tests
            ["-m", "unit"],
            # Run only fast tests (assuming marker exists)
            ["-m", "not slow"],
            # Run tests by keyword
            ["-k", "config"],
            # Run specific test file
            ["tests/unit/test_config_manager.py"],
        ]

        for strategy in strategies:
            result = _run_command(
                [sys.executable, "-m", "pytest", *strategy, "--collect-only", "-q"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            # Each strategy should work (may collect 0 tests)
            assert result.returncode in [0, 5]  # pytest uses 5 for an empty selection

    def test_test_matrix_execution(self):
        """Test execution across different configurations"""
        # Simulate testing with different Python versions/configurations
        configurations = [
            {"TEST_MODE": "fast"},
            {"TEST_MODE": "thorough"},
            {"DEBUG": "true"},
            {"PYTHONPATH": str(Path.cwd())},
        ]

        for config in configurations:
            env = os.environ.copy()
            env.update(config)

            result = _run_command(
                [sys.executable, "-m", "pytest", "tests/unit/test_config_manager.py", "-q"],
                capture_output=True,
                text=True,
                env=env,
                cwd=Path.cwd(),
            )

            # Should handle different configurations
            assert result.returncode == 0, result.stdout + result.stderr


class TestDocumentationTesting:
    """Test documentation testing integration"""

    def test_docstring_testing(self):
        """Test that docstring examples are tested"""
        try:
            # Test doctest integration
            result = _run_command(
                [sys.executable, "-m", "pytest", "--doctest-modules", "core/config_manager.py"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            # Should execute without errors (even if no doctests found)
            assert result.returncode in [0, 5]  # No examples is distinct from collection/test failure

        except subprocess.TimeoutExpired:
            pytest.fail("Doctest subprocess exceeded its bounded timeout")

    def test_readme_code_validation(self):
        """Check documented maintained entry and parse any Python snippets."""
        import ast
        import re

        content = Path("README.md").read_text()
        assert "servers.mcp_server_sdk:cli_main" in content
        for snippet in re.findall(r"```python\s*\n(.*?)```", content, re.DOTALL):
            ast.parse(snippet)
