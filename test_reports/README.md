# Test Reports Directory Structure

This directory contains organized test reports for the MCP Academic RAG Server project.

## Directory Structure

```
test_reports/
├── unit/              # Unit test reports and coverage
├── integration/       # Integration test reports
├── e2e/              # End-to-end test reports
├── performance/       # Performance test benchmarks
├── coverage/         # Code coverage reports
└── latest/           # Latest test results (symlinks for quick access)
```

## Report Types

### Unit Tests (`unit/`)
- Individual component test results
- Code coverage reports for modules
- Test execution times and statistics

### Integration Tests (`integration/`)
- Multi-component integration test results
- System workflow validation reports
- API endpoint testing results

### End-to-End Tests (`e2e/`)
- Complete user workflow test results
- Full system behavior validation
- User acceptance test reports

### Performance Tests (`performance/`)
- Benchmark test results
- Performance regression reports
- Load testing results
- Resource usage analysis

### Coverage Reports (`coverage/`)
- Overall project code coverage
- Coverage trends over time
- Detailed coverage by module

### Latest Results (`latest/`)
- Quick access to most recent test results
- Consolidated summary reports
- Build status indicators

## Report Formats

- **HTML**: Interactive coverage and test reports
- **JSON**: Machine-readable test results for CI/CD
- **Markdown**: Human-readable summary reports
- **XML**: JUnit-compatible test results

## Automated Report Generation

Test reports are automatically generated during:
- Local test execution (`pytest --cov`)
- CI/CD pipeline runs
- Performance benchmarking
- Code quality checks

## Retention Policy

- **Latest**: Always available
- **Weekly**: Keep for 4 weeks
- **Monthly**: Keep for 6 months
- **Release**: Keep permanently

## Accessing Reports

```bash
# Run tests with coverage
pytest --cov=. --cov-report=html:test_reports/coverage/

# Performance benchmarks
python -m pytest tests/performance/ --benchmark-only

# Generate comprehensive report
python scripts/run_tests.py --full-report
```

For CI/CD integration, reports are automatically published to GitHub Pages and accessible via the project dashboard.