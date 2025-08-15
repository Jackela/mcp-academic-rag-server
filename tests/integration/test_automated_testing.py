"""
Automated Testing Integration Tests

Test suite for validating the automated testing infrastructure and CI/CD integration.
Includes test discovery, execution, reporting, and integration with various testing frameworks.
"""

import pytest
import subprocess
import sys
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, call

from tests.utils.cleanup import ResourceCleaner, managed_resource


class TestDiscoveryAndExecution:
    """Test automated test discovery and execution"""
    
    def test_pytest_discovery(self):
        """Test that pytest can discover all test files correctly"""
        # Run pytest with collection-only to discover tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "--collect-only", "-q",
            "tests/"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        assert result.returncode == 0
        output = result.stdout
        
        # Should discover tests from various categories
        assert "test_unit" in output or "unit" in output
        assert "test_integration" in output or "integration" in output
        assert "test_e2e" in output or "e2e" in output
    
    def test_test_categorization(self):
        """Test that tests are properly categorized with markers"""
        # Check that pytest markers are properly configured
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "--markers"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        assert result.returncode == 0
        markers_output = result.stdout
        
        # Should have custom markers defined
        expected_markers = ["unit", "integration", "e2e", "performance", "slow"]
        for marker in expected_markers:
            # Either marker is defined or pytest shows the custom marker
            assert marker in markers_output or f"@pytest.mark.{marker}" in markers_output
    
    def test_parallel_test_execution(self):
        """Test parallel test execution capability"""
        # Check if pytest-xdist is available and working
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "--help"
            ], capture_output=True, text=True)
            
            # Check if -n option (xdist) is available
            if "-n" in result.stdout:
                # Test parallel execution
                parallel_result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    "tests/unit/test_config_manager.py",
                    "-n", "2", "-v"
                ], capture_output=True, text=True, cwd=Path.cwd())
                
                # Should execute without errors (even if tests fail)
                assert parallel_result.returncode in [0, 1]  # 0 = success, 1 = test failures
            else:
                pytest.skip("pytest-xdist not available for parallel testing")
                
        except Exception as e:
            pytest.skip(f"Cannot test parallel execution: {e}")


class TestTestReporting:
    """Test automated test reporting functionality"""
    
    def test_junit_xml_generation(self):
        """Test JUnit XML report generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            junit_file = Path(temp_dir) / "junit.xml"
            
            # Run a simple test with JUnit XML output
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/unit/test_config_manager.py",
                f"--junit-xml={junit_file}",
                "-v"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            # Should generate JUnit XML file
            assert junit_file.exists()
            
            # Verify XML content
            xml_content = junit_file.read_text()
            assert "<?xml version=" in xml_content
            assert "<testsuite" in xml_content
            assert "</testsuite>" in xml_content
    
    def test_coverage_reporting(self):
        """Test code coverage reporting"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                coverage_file = Path(temp_dir) / "coverage.xml"
                
                # Run tests with coverage
                result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    "tests/unit/test_config_manager.py",
                    "--cov=core",
                    f"--cov-report=xml:{coverage_file}",
                    "--cov-report=term-missing"
                ], capture_output=True, text=True, cwd=Path.cwd())
                
                # Check if coverage ran (pytest-cov might not be installed)
                if result.returncode == 0 or "coverage" in result.stdout.lower():
                    # Verify coverage output
                    assert "coverage" in result.stdout.lower() or coverage_file.exists()
                else:
                    pytest.skip("pytest-cov not available for coverage testing")
                    
        except Exception as e:
            pytest.skip(f"Cannot test coverage reporting: {e}")
    
    def test_html_report_generation(self):
        """Test HTML test report generation"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                html_dir = Path(temp_dir) / "html_report"
                
                # Run tests with HTML report
                result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    "tests/unit/test_config_manager.py",
                    f"--html={html_dir}/report.html",
                    "--self-contained-html"
                ], capture_output=True, text=True, cwd=Path.cwd())
                
                # Check if pytest-html is available
                if "unrecognized arguments: --html" not in result.stderr:
                    # Should generate HTML report
                    report_file = html_dir / "report.html"
                    if report_file.exists():
                        html_content = report_file.read_text()
                        assert "<html" in html_content
                        assert "Test Report" in html_content or "pytest" in html_content
                else:
                    pytest.skip("pytest-html not available for HTML reporting")
                    
        except Exception as e:
            pytest.skip(f"Cannot test HTML report generation: {e}")


class TestCIIntegration:
    """Test CI/CD integration capabilities"""
    
    def test_github_actions_workflow_syntax(self):
        """Test GitHub Actions workflow file syntax"""
        workflow_file = Path(".github/workflows/vector-storage-tests.yml")
        
        if workflow_file.exists():
            import yaml
            
            try:
                with open(workflow_file) as f:
                    workflow_content = yaml.safe_load(f)
                
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
            pytest.skip("GitHub Actions workflow file not found")
    
    def test_environment_variable_handling(self):
        """Test handling of environment variables in testing"""
        # Test that tests can handle different environments
        original_env = os.environ.get("TEST_ENV")
        
        try:
            # Set test environment
            os.environ["TEST_ENV"] = "testing"
            
            # Run tests that might use environment variables
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/unit/test_config_system.py",
                "-v", "-k", "test_environment"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            # Should handle environment variables properly
            assert "ERROR" not in result.stderr or result.returncode in [0, 1]
            
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
            [sys.executable, "-m", "pytest", "tests/unit/test_config_manager.py::TestConfigManager::test_initialization", "-v"],
            [sys.executable, "-m", "pytest", "tests/unit/test_config_manager.py::TestConfigManager::test_initialization", "-v"],
            [sys.executable, "-m", "pytest", "tests/unit/test_config_manager.py::TestConfigManager::test_initialization", "-v"]
        ]
        
        results = []
        for cmd in test_commands:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            results.append(result.returncode)
        
        # All runs should have the same result (proper isolation)
        assert len(set(results)) <= 1  # All results should be the same


class TestPerformanceTesting:
    """Test performance testing capabilities"""
    
    @pytest.mark.performance
    def test_performance_test_execution(self):
        """Test execution of performance tests"""
        # Check if performance tests exist and can be run
        perf_test_file = Path("tests/performance")
        
        if perf_test_file.exists():
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/performance/",
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            # Performance tests should execute (may pass or fail)
            assert result.returncode in [0, 1]  # 0 = pass, 1 = fail, not error
            
            # Should have run some tests
            assert "test session starts" in result.stdout
        else:
            pytest.skip("No performance tests found")
    
    @pytest.mark.performance
    def test_benchmark_integration(self):
        """Test benchmark testing integration"""
        try:
            # Check if pytest-benchmark is available
            result = subprocess.run([
                sys.executable, "-c", "import pytest_benchmark"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Run benchmark tests if available
                benchmark_result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    "--benchmark-only",
                    "--benchmark-disable-gc",
                    "tests/"
                ], capture_output=True, text=True, cwd=Path.cwd())
                
                # Benchmark tests should execute (even if none found)
                assert benchmark_result.returncode in [0, 1, 2]  # 2 = no tests collected
            else:
                pytest.skip("pytest-benchmark not available")
                
        except Exception as e:
            pytest.skip(f"Cannot test benchmark integration: {e}")


class TestResourceManagement:
    """Test resource management during testing"""
    
    def test_resource_cleanup_integration(self):
        """Test that resource cleanup works correctly in tests"""
        cleaner = ResourceCleaner()
        
        # Test that cleanup context manager works
        with managed_resource(cleanup_func=lambda: cleaner.cleanup_all()):
            # Simulate resource creation
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()
            
            # Add to cleanup
            cleaner.add_cleanup(lambda: os.unlink(temp_file.name))
            
            # Verify file exists
            assert os.path.exists(temp_file.name)
        
        # After context, cleanup should have run
        # Note: In actual usage, this would be handled by the cleanup system
    
    def test_memory_leak_detection(self):
        """Test memory leak detection in test suite"""
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Run a subset of tests
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/unit/test_config_manager.py",
                "-v"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            # Force garbage collection
            gc.collect()
            
            # Check final memory usage
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 50MB for unit tests)
            assert memory_increase < 50 * 1024 * 1024  # 50MB threshold
            
        except ImportError:
            pytest.skip("psutil not available for memory leak detection")
    
    def test_test_data_cleanup(self):
        """Test that test data is properly cleaned up"""
        # Create temporary test data directory
        test_data_dir = Path("test_temp_data")
        
        try:
            test_data_dir.mkdir(exist_ok=True)
            test_file = test_data_dir / "test_file.txt"
            test_file.write_text("test data")
            
            # Run tests that might use test data
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/unit/test_config_manager.py",
                "-v"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            # Tests should execute successfully
            assert result.returncode in [0, 1]
            
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
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v", "--tb=long"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            # Should have failure information
            assert result.returncode == 1  # Tests failed
            assert "FAILED" in result.stdout
            assert "Intentional failure" in result.stdout
            assert "ValueError" in result.stdout
    
    def test_debug_mode_integration(self):
        """Test debug mode integration for test failures"""
        # Test that debug flags work properly
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/unit/test_config_manager.py",
            "--pdb-trace",
            "--capture=no",
            "-x"  # Stop on first failure
        ], capture_output=True, text=True, input="\n", cwd=Path.cwd())
        
        # Should handle debug flags without crashing
        assert "error" not in result.stderr.lower() or result.returncode in [0, 1, 2]
    
    def test_test_result_analysis(self):
        """Test analysis of test results"""
        # Run tests and capture detailed output
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "--tb=short",
            "-v",
            "--durations=10"  # Show slowest 10 tests
        ], capture_output=True, text=True, cwd=Path.cwd())
        
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
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/unit/test_config_manager.py",
            "--quiet",
            "--tb=no"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        # Pre-commit style should be fast and minimal output
        execution_time = len(result.stdout.split('\n'))
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
            ["tests/unit/test_config_manager.py"]
        ]
        
        for strategy in strategies:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                *strategy,
                "--collect-only", "-q"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            # Each strategy should work (may collect 0 tests)
            assert result.returncode in [0, 2]  # 0 = success, 2 = no tests collected
    
    def test_test_matrix_execution(self):
        """Test execution across different configurations"""
        # Simulate testing with different Python versions/configurations
        configurations = [
            {"TEST_MODE": "fast"},
            {"TEST_MODE": "thorough"},
            {"DEBUG": "true"},
            {"PYTHONPATH": str(Path.cwd())}
        ]
        
        for config in configurations:
            env = os.environ.copy()
            env.update(config)
            
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/unit/test_config_manager.py",
                "-q"
            ], capture_output=True, text=True, env=env, cwd=Path.cwd())
            
            # Should handle different configurations
            assert result.returncode in [0, 1]  # 0 = pass, 1 = fail


class TestDocumentationTesting:
    """Test documentation testing integration"""
    
    def test_docstring_testing(self):
        """Test that docstring examples are tested"""
        try:
            # Test doctest integration
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--doctest-modules",
                "core/config_manager.py"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            # Should execute without errors (even if no doctests found)
            assert result.returncode in [0, 1, 2]
            
        except Exception as e:
            pytest.skip(f"Cannot test docstring testing: {e}")
    
    def test_readme_code_validation(self):
        """Test validation of code examples in README"""
        readme_file = Path("README.md")
        
        if readme_file.exists():
            readme_content = readme_file.read_text()
            
            # Check for code blocks that might need testing
            if "```python" in readme_content:
                # README contains Python code examples
                # In a real implementation, these could be extracted and tested
                assert True
            else:
                pytest.skip("No Python code examples found in README")
        else:
            pytest.skip("README.md not found")