# Developer Guide

🎆 **Status**: PRODUCTION READY - Fully deployed and operational  
🚀 **Architecture**: Refocused from enterprise platform to streamlined MCP server  
📊 **Dependencies**: Reduced from 70+ to 14 core packages (80% reduction)  
📄 **Documentation**: Updated to professional open source standards  

This guide provides comprehensive information for developers working with the production-ready MCP Academic RAG Server codebase. It covers the refocused architecture, development patterns, testing strategies, and contribution guidelines.

## Architecture Overview

### 🏠 Refocused System Architecture

The server implements a **streamlined, production-ready architecture** focused on core MCP functionality. This represents a successful refocusing from an over-engineered enterprise platform to a reliable, maintainable MCP server:

**Before Refocusing**: 7 Docker services, 70+ dependencies, Kubernetes manifests, complex monitoring  
**After Refocusing**: Single service, 14 dependencies, local storage, focused MCP tools  

### System Components

```
MCP Academic RAG Server
├── mcp_server.py                 # MCP protocol interface
├── core/                         # Core system components
│   ├── config_manager.py         # Configuration management
│   └── pipeline.py               # Document processing pipeline
├── processors/                   # Document processing modules
│   ├── base_processor.py         # Abstract processor interface
│   └── ocr_processor.py          # OCR implementation
├── rag/                          # RAG system components
│   ├── haystack_pipeline.py      # RAG pipeline implementation
│   └── chat_session.py           # Session management
├── models/                       # Data models
│   ├── document.py               # Document data model
│   └── process_result.py         # Processing result model
└── utils/                        # Utility modules
    └── vector_utils.py           # Vector operations
```

### Data Flow Architecture

```
Input Document → Document Model → Processing Pipeline → Vector Storage
                                                            ↓
Query Request → Session Context → RAG Pipeline → Response Generation
```

### Design Principles

The codebase follows these core principles:

- **Modularity**: Components are loosely coupled with clear interfaces
- **Testability**: Each component can be unit tested in isolation  
- **Configurability**: Behavior is controlled through configuration files
- **Error Handling**: Comprehensive error handling with detailed logging
- **Protocol Compliance**: Strict adherence to MCP specification

## Core Interfaces

### MCP Server Interface

The main server interface implements the MCP protocol:

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("academic-rag-server")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """Return available MCP tools."""
    return [
        Tool(
            name="process_document",
            description="Process academic documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "file_name": {"type": "string"}
                },
                "required": ["file_path"]
            }
        ),
        # Additional tools...
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle MCP tool invocations."""
    if name == "process_document":
        return await process_document(arguments)
    # Handle other tools...
```

### Processor Interface

All document processors implement the `IProcessor` interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class IProcessor(ABC):
    """Abstract interface for document processors."""
    
    @abstractmethod
    async def process(self, document: Document) -> ProcessResult:
        """
        Process a document and return results.
        
        Args:
            document: Document instance to process
            
        Returns:
            ProcessResult indicating success/failure with details
        """
        pass
        
    def get_name(self) -> str:
        """Return processor name for identification."""
        return self.__class__.__name__
```

### Data Models

#### Document Model

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid

@dataclass
class Document:
    """Document data model for processing pipeline."""
    
    file_path: str
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_name: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.file_name is None:
            self.file_name = os.path.basename(self.file_path)
```

#### Processing Result Model

```python
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class ProcessResult:
    """Result of document processing operation."""
    
    success: bool
    message: str = ""
    error: Optional[Exception] = None
    data: Optional[Any] = None
    
    def is_successful(self) -> bool:
        """Check if processing succeeded."""
        return self.success and self.error is None
```

## Development Patterns

### Processor Implementation

When creating new processors, follow this pattern:

```python
from processors.base_processor import IProcessor
from models.process_result import ProcessResult

class CustomProcessor(IProcessor):
    """Example custom processor implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration parameters."""
        self.config = config
        self.enabled = config.get("enabled", True)
    
    async def process(self, document: Document) -> ProcessResult:
        """Process document with custom logic."""
        if not self.enabled:
            return ProcessResult(
                success=True,
                message=f"{self.get_name()} skipped (disabled)"
            )
        
        try:
            # Implement processing logic
            result_data = await self._execute_processing(document)
            
            # Update document with results
            document.content[self.get_name()] = result_data
            
            return ProcessResult(
                success=True,
                message=f"{self.get_name()} completed successfully",
                data=result_data
            )
            
        except Exception as e:
            return ProcessResult(
                success=False,
                message=f"{self.get_name()} failed: {str(e)}",
                error=e
            )
    
    async def _execute_processing(self, document: Document) -> Dict[str, Any]:
        """Execute core processing logic."""
        # Implement specific processing steps
        return {"processed": True, "timestamp": time.time()}
```

### Error Handling Patterns

Implement consistent error handling across components:

```python
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

def handle_processing_errors(operation_name: str):
    """Decorator for consistent error handling."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except FileNotFoundError as e:
                logger.error(f"{operation_name}: File not found - {e}")
                raise ProcessingError(f"File not found: {e}")
            except PermissionError as e:
                logger.error(f"{operation_name}: Permission denied - {e}")
                raise ProcessingError(f"Permission denied: {e}")
            except Exception as e:
                logger.error(f"{operation_name}: Unexpected error - {e}")
                raise ProcessingError(f"Processing failed: {e}")
        return wrapper
    return decorator

# Usage example
class OCRProcessor(IProcessor):
    @handle_processing_errors("OCR processing")
    async def process(self, document: Document) -> ProcessResult:
        # Processing implementation
        pass
```

### Configuration Management

Access configuration through the centralized manager:

```python
from core.config_manager import ConfigManager

class ComponentWithConfig:
    """Example component using configuration."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get configuration setting with fallback."""
        try:
            return self.config_manager.get(key)
        except KeyError:
            return default
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if feature is enabled in configuration."""
        return self.get_setting(f"features.{feature}", False)
```

### Async Processing Patterns

Handle asynchronous operations properly:

```python
import asyncio
from typing import List, Coroutine

async def process_documents_concurrently(
    documents: List[Document], 
    max_concurrent: int = 3
) -> List[ProcessResult]:
    """Process multiple documents with concurrency limits."""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(doc: Document) -> ProcessResult:
        async with semaphore:
            return await process_single_document(doc)
    
    # Create tasks for all documents
    tasks = [process_with_semaphore(doc) for doc in documents]
    
    # Execute with error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to error results
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(ProcessResult(
                success=False,
                error=result,
                message=f"Processing failed: {str(result)}"
            ))
        else:
            processed_results.append(result)
    
    return processed_results
```

## Testing Strategy

### Unit Testing Framework

The project uses pytest with additional plugins for async testing:

```python
# tests/conftest.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "server": {"port": 8000, "name": "test-server"},
        "storage": {"type": "memory", "path": "/tmp/test"},
        "processing": {"ocr": {"enabled": True}}
    }

@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return Document(
        file_path="/test/sample.pdf",
        file_name="sample.pdf"
    )

@pytest.fixture
def mock_processor():
    """Mock processor for testing."""
    processor = Mock(spec=IProcessor)
    processor.get_name.return_value = "MockProcessor"
    processor.process = AsyncMock(return_value=ProcessResult(
        success=True,
        message="Mock processing completed"
    ))
    return processor
```

### Test Categories

#### Unit Tests

Test individual components in isolation:

```python
# tests/unit/test_processors.py
import pytest
from processors.ocr_processor import OCRProcessor
from models.document import Document

class TestOCRProcessor:
    """Test OCR processor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"enabled": True, "language": "eng"}
        self.processor = OCRProcessor(self.config)
    
    @pytest.mark.asyncio
    async def test_process_success(self, sample_document):
        """Test successful document processing."""
        result = await self.processor.process(sample_document)
        
        assert result.is_successful()
        assert self.processor.get_name() in sample_document.content
        assert result.message == "OCRProcessor completed successfully"
    
    @pytest.mark.asyncio
    async def test_process_file_not_found(self):
        """Test handling of missing files."""
        document = Document(file_path="/nonexistent/file.pdf")
        result = await self.processor.process(document)
        
        assert not result.is_successful()
        assert "not found" in result.message.lower()
        assert isinstance(result.error, FileNotFoundError)
    
    def test_processor_name(self):
        """Test processor name retrieval."""
        assert self.processor.get_name() == "OCRProcessor"
```

#### Integration Tests

Test component interactions:

```python
# tests/integration/test_pipeline.py
import pytest
from core.pipeline import Pipeline
from models.document import Document

class TestProcessingPipeline:
    """Test document processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_config, tmp_path):
        """Test complete document processing pipeline."""
        # Create test document
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")
        
        document = Document(file_path=str(test_file))
        pipeline = Pipeline(mock_config)
        
        # Execute pipeline
        result = await pipeline.process_document(document)
        
        # Verify results
        assert result.is_successful()
        assert document.content  # Should have processing results
        assert "processing_time" in document.metadata
```

#### MCP Tool Tests

Test MCP protocol compliance:

```python
# tests/integration/test_mcp_tools.py
import pytest
import json
from mcp_server import process_document, query_documents

class TestMCPTools:
    """Test MCP tool implementations."""
    
    @pytest.mark.asyncio
    async def test_process_document_tool(self, tmp_path):
        """Test process_document MCP tool."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"Test content")
        
        # Call tool
        arguments = {"file_path": str(test_file)}
        result = await process_document(arguments)
        
        # Verify MCP response format
        assert len(result) == 1
        assert result[0].type == "text"
        
        # Parse response
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        assert "document_id" in response_data
    
    @pytest.mark.asyncio
    async def test_query_documents_tool(self):
        """Test query_documents MCP tool."""
        arguments = {
            "query": "What is machine learning?",
            "top_k": 3
        }
        
        result = await query_documents(arguments)
        
        # Verify response structure
        assert len(result) == 1
        response_data = json.loads(result[0].text)
        assert "answer" in response_data
        assert "sources" in response_data
```

### Test Execution

Run tests with appropriate coverage reporting:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/                 # Unit tests only
pytest tests/integration/          # Integration tests only
pytest -m "not slow"               # Skip slow tests

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_processors.py
```

## Deployment and Operations

### Development Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install

# Start development server
python mcp_server.py
```

### Production Deployment

#### Using uvx

```bash
# Production deployment
uvx --from . mcp-academic-rag-server

# With custom configuration
CONFIG_PATH=/etc/mcp/config.json uvx --from . mcp-academic-rag-server
```

#### Using Docker

```bash
# Build production image
docker build -f Dockerfile.simple -t mcp-academic-rag:latest .

# Run container
docker run -d \
  --name mcp-academic-rag \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -e OPENAI_API_KEY=your_key \
  mcp-academic-rag:latest
```

### Monitoring and Observability

#### Health Checks

Implement health check endpoints:

```python
# health_check.py
import asyncio
import logging
from typing import Dict, Any

async def check_system_health() -> Dict[str, Any]:
    """Perform comprehensive system health check."""
    checks = {
        "mcp_server": await check_mcp_server(),
        "document_storage": await check_document_storage(),
        "vector_index": await check_vector_index(),
        "external_apis": await check_external_apis()
    }
    
    overall_status = "healthy" if all(
        check["status"] == "healthy" for check in checks.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }
```

#### Logging Configuration

Configure structured logging:

```python
# utils/logging_config.py
import logging
import sys
from loguru import logger

def configure_logging(level: str = "INFO"):
    """Configure application logging."""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Add file handler for persistent logging
    logger.add(
        "logs/mcp-server.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=level,
        rotation="10 MB",
        retention="30 days"
    )
```

## Contributing Guidelines

### Development Workflow

1. **Fork Repository**: Create personal fork of the main repository
2. **Create Branch**: Create feature branch with descriptive name
3. **Development**: Implement changes following coding standards
4. **Testing**: Ensure all tests pass and add new tests for new functionality
5. **Documentation**: Update relevant documentation
6. **Pull Request**: Submit PR with clear description of changes

### Code Standards

#### Python Code Style

Follow PEP 8 with project-specific extensions:

```python
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
```

#### Documentation Standards

- Use docstrings for all public functions and classes
- Include type hints for function signatures
- Provide usage examples for complex functionality
- Update API documentation for interface changes

#### Commit Message Format

Follow conventional commit format:

```
type(scope): brief description

Longer description explaining the change, its motivation,
and any breaking changes.

Fixes #123
```

### Testing Requirements

- Maintain minimum 80% test coverage
- Add unit tests for all new functionality
- Include integration tests for API changes
- Test error handling and edge cases
- Verify MCP protocol compliance

### Performance Considerations

#### Memory Management

Monitor memory usage in processing pipelines:

```python
import psutil
import gc
from typing import Any, Callable

def monitor_memory_usage(func: Callable) -> Callable:
    """Decorator to monitor memory usage."""
    def wrapper(*args, **kwargs) -> Any:
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Check final memory usage
            final_memory = process.memory_info().rss
            memory_delta = final_memory - initial_memory
            
            if memory_delta > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"High memory usage: {memory_delta / 1024 / 1024:.1f}MB")
            
            # Force garbage collection for large memory increases
            if memory_delta > 500 * 1024 * 1024:  # 500MB
                gc.collect()
    
    return wrapper
```

#### Async Operation Optimization

Use appropriate concurrency patterns:

```python
import asyncio
from typing import List, TypeVar, Awaitable

T = TypeVar('T')

async def process_with_backpressure(
    items: List[T],
    processor: Callable[[T], Awaitable[Any]],
    max_concurrent: int = 10,
    batch_size: int = 100
) -> List[Any]:
    """Process items with backpressure control."""
    
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        async def process_with_semaphore(item: T) -> Any:
            async with semaphore:
                return await processor(item)
        
        batch_tasks = [process_with_semaphore(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)
        
        # Allow other coroutines to run between batches
        await asyncio.sleep(0)
    
    return results
```

This developer guide provides the foundation for contributing to the MCP Academic RAG Server codebase while maintaining code quality, performance, and protocol compliance standards.