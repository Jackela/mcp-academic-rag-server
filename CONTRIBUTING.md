# Contributing to MCP Academic RAG Server

Thank you for your interest in contributing to the MCP Academic RAG Server! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully
- Prioritize the project's best interests

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mcp-academic-rag-server.git
   cd mcp-academic-rag-server
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original/mcp-academic-rag-server.git
   ```

4. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (optional, for containerized development)
- Git

### Local Development

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   # Install all dependencies including dev tools
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   cp config/config.json.example config/config.json
   # Edit .env and config.json with your settings
   ```

5. **Run tests to verify setup**:
   ```bash
   pytest tests/unit/
   ```

### Docker Development

1. **Build development image**:
   ```bash
   docker-compose -f docker-compose.dev.yml build
   ```

2. **Start services**:
   ```bash
   docker-compose -f docker-compose.dev.yml up
   ```

3. **Run tests in container**:
   ```bash
   docker-compose -f docker-compose.dev.yml run --rm academic-rag-server pytest
   ```

## How to Contribute

### Reporting Issues

1. **Check existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Error messages and logs

### Suggesting Features

1. **Open a discussion** first for major features
2. **Provide use cases** and examples
3. **Consider implementation complexity**
4. **Be open to feedback** and alternative approaches

### Fixing Bugs

1. **Claim an issue** by commenting on it
2. **Write tests** that reproduce the bug
3. **Fix the bug** with minimal changes
4. **Ensure all tests pass**

### Adding Features

1. **Discuss first** via issue or discussion
2. **Design the feature** with maintainers
3. **Implement incrementally** with tests
4. **Update documentation**
5. **Add examples** if applicable

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 120 characters maximum
- **Imports**: Use `isort` for ordering
- **Formatting**: Use `black` for consistency
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public APIs

### Code Quality Tools

Run these before committing:

```bash
# Format code
black .
isort .

# Check code quality
flake8 .
mypy .

# Security scan
bandit -r .
safety check
```

### Best Practices

1. **Single Responsibility**: Each function/class should do one thing
2. **DRY (Don't Repeat Yourself)**: Extract common code
3. **SOLID Principles**: Follow OOP best practices
4. **Error Handling**: Use custom exceptions and proper logging
5. **Performance**: Profile before optimizing

### Example Code Style

```python
"""Module docstring describing the purpose."""

import logging
from typing import List, Optional, Dict, Any

from utils.error_handling import ProcessingError

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process academic documents.
    
    This class handles the processing of various document formats
    for the academic RAG system.
    
    Attributes:
        config: Configuration dictionary
        processors: List of processor instances
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the document processor.
        
        Args:
            config: Configuration dictionary containing processor settings
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config
        self.processors: List[IProcessor] = []
        self._setup_processors()
    
    def process_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessResult:
        """
        Process a single document.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata for the document
            
        Returns:
            ProcessResult containing the processing outcome
            
        Raises:
            ProcessingError: If document processing fails
        """
        try:
            # Implementation here
            pass
        except Exception as e:
            logger.error(f"Failed to process document: {file_path}", exc_info=True)
            raise ProcessingError(f"Processing failed: {str(e)}") from e
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── e2e/           # End-to-end tests for complete workflows
└── performance/   # Performance and benchmark tests
```

### Writing Tests

1. **Test naming**: `test_<function_name>_<scenario>`
2. **Use pytest fixtures** for common setup
3. **Mock external dependencies**
4. **Test edge cases** and error conditions
5. **Aim for 80%+ coverage**

### Example Test

```python
"""Tests for document processor."""

import pytest
from unittest.mock import Mock, patch

from processors.document_processor import DocumentProcessor
from utils.error_handling import ProcessingError


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance for testing."""
        config = {"key": "value"}
        return DocumentProcessor(config)
    
    def test_process_document_success(self, processor, tmp_path):
        """Test successful document processing."""
        # Arrange
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")
        
        # Act
        result = processor.process_document(str(test_file))
        
        # Assert
        assert result.is_successful()
        assert result.document_id is not None
    
    def test_process_document_file_not_found(self, processor):
        """Test processing with non-existent file."""
        # Act & Assert
        with pytest.raises(ProcessingError, match="File not found"):
            processor.process_document("/non/existent/file.pdf")
    
    @patch('processors.document_processor.external_api')
    def test_process_document_api_failure(self, mock_api, processor, tmp_path):
        """Test handling of external API failures."""
        # Arrange
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")
        mock_api.side_effect = Exception("API error")
        
        # Act & Assert
        with pytest.raises(ProcessingError, match="API error"):
            processor.process_document(str(test_file))
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_document_processor.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run only marked tests
pytest -m unit
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto
```

## Documentation

### Code Documentation

1. **Module docstrings**: Describe the module's purpose
2. **Class docstrings**: Explain the class and its attributes
3. **Function docstrings**: Document parameters, returns, and exceptions
4. **Inline comments**: Explain complex logic
5. **Type hints**: For all function signatures

### User Documentation

Update relevant documentation when making changes:

- `README.md`: Project overview and quick start
- `docs/`: Detailed documentation
- `examples/`: Working examples
- API documentation: OpenAPI/Swagger specs

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

Examples:
```
feat(processor): add support for DOCX files

- Implement DOCX parser using python-docx
- Add tests for DOCX processing
- Update documentation

Closes #123
```

## Submitting Changes

### Before Submitting

1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Format code
   black . && isort .
   
   # Run tests
   pytest
   
   # Check code quality
   flake8 . && mypy .
   
   # Security check
   bandit -r . && safety check
   ```

3. **Update documentation** if needed

4. **Add/update tests** for your changes

5. **Check test coverage**:
   ```bash
   pytest --cov=. --cov-report=term-missing
   ```

### Creating a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR** via GitHub UI

3. **Fill out PR template** completely

4. **Link related issues** using keywords like "Fixes #123"

5. **Request reviews** from maintainers

### PR Guidelines

- **Keep PRs focused**: One feature/fix per PR
- **Write descriptive titles**: Summarize the change
- **Provide context**: Explain why the change is needed
- **Include tests**: All new code needs tests
- **Update docs**: Keep documentation in sync
- **Be responsive**: Address review feedback promptly

## Review Process

### What to Expect

1. **Automated checks** run first (CI/CD)
2. **Code review** by maintainers
3. **Discussion** and feedback
4. **Revisions** if needed
5. **Approval** and merge

### Review Criteria

- **Code quality**: Follows standards and best practices
- **Tests**: Adequate coverage and passing
- **Documentation**: Updated as needed
- **Performance**: No significant regressions
- **Security**: No vulnerabilities introduced
- **Design**: Fits project architecture

### After Merge

1. **Delete your branch** (GitHub can do this automatically)
2. **Update your fork**:
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```
3. **Celebrate!** 🎉 Thank you for contributing!

## Getting Help

- **Discord/Slack**: Join our community chat
- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs or request features
- **Email**: contact@academic-rag.com

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for helping make MCP Academic RAG Server better!