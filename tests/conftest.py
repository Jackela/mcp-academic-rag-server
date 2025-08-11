"""
Enhanced pytest configuration for MCP Academic RAG Server

This module provides improved fixtures and test utilities including:
- Async test support
- Performance testing utilities
- Mock API service providers
- Test data generation
- Database test fixtures
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import logging
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json
from datetime import datetime

# Import project modules
from models.document import Document
from models.process_result import ProcessResult
from core.config_manager import ConfigManager
from core.pipeline import Pipeline
from processors.base_processor import BaseProcessor
from rag.chat_session import ChatSession, ChatSessionManager
from utils.performance_enhancements import MemoryManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="mcp_rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_test_dir) -> Dict[str, Any]:
    """Enhanced test configuration with comprehensive settings."""
    return {
        "storage": {
            "base_path": os.path.join(temp_test_dir, "data"),
            "output_path": os.path.join(temp_test_dir, "output"),
            "temp_path": os.path.join(temp_test_dir, "temp")
        },
        "processors": {
            "pre_processor": {
                "enabled": True,
                "config": {
                    "enhance_image": False,  # Disable for faster tests
                    "correct_skew": False
                }
            },
            "ocr_processor": {
                "enabled": True,
                "config": {
                    "api": "mock",  # Use mock API for tests
                    "language": "en"
                }
            },
            "embedding_processor": {
                "enabled": True,
                "config": {
                    "model_name": "mock-embedding-model",
                    "chunk_size": 100,  # Smaller chunks for faster tests
                    "chunk_overlap": 10,
                    "batch_size": 4
                }
            }
        },
        "vector_db": {
            "document_store": {
                "type": "memory",  # Use in-memory store for tests
                "embedding_dim": 128,  # Smaller dimensions for tests
                "similarity": "cosine"
            }
        },
        "llm": {
            "type": "mock",
            "model": "mock-gpt-3.5-turbo",
            "settings": {
                "temperature": 0.0,  # Deterministic for tests
                "max_tokens": 100
            }
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
def config_manager(test_config, temp_test_dir):
    """Create a ConfigManager instance for tests."""
    config_file = os.path.join(temp_test_dir, "test_config.json")
    with open(config_file, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    return ConfigManager(config_file)


@pytest.fixture
def sample_document(temp_test_dir) -> Document:
    """Create a sample document for testing."""
    # Create a sample PDF file
    sample_file = os.path.join(temp_test_dir, "sample_document.pdf")
    with open(sample_file, 'wb') as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
    
    document = Document(
        file_path=sample_file,
        file_name="sample_document.pdf",
        file_type="pdf"
    )
    document.add_metadata("author", "Test Author")
    document.add_metadata("title", "Sample Test Document")
    document.add_tag("test")
    document.add_tag("sample")
    
    return document


@pytest.fixture
def sample_documents(temp_test_dir) -> List[Document]:
    """Create multiple sample documents for batch testing."""
    documents = []
    
    for i in range(5):
        file_path = os.path.join(temp_test_dir, f"test_doc_{i}.pdf")
        with open(file_path, 'wb') as f:
            f.write(f"%PDF-1.4\nTest Document {i} Content".encode())
        
        doc = Document(
            file_path=file_path,
            file_name=f"test_doc_{i}.pdf",
            file_type="pdf"
        )
        doc.add_metadata("doc_index", i)
        doc.add_tag(f"doc_{i}")
        documents.append(doc)
    
    return documents


@pytest.fixture
def mock_processor():
    """Create a mock processor for testing."""
    class MockProcessor(BaseProcessor):
        def __init__(self, name="MockProcessor", should_fail=False):
            super().__init__(name, "Mock processor for testing")
            self.should_fail = should_fail
            self.process_count = 0
        
        def process(self, document: Document) -> ProcessResult:
            self.process_count += 1
            
            if self.should_fail:
                return ProcessResult.error_result(
                    "Mock processor intentionally failed",
                    Exception("Mock error")
                )
            
            # Simulate processing by adding content
            document.store_content(
                "mock_processing",
                {"processed": True, "processor": self.get_name(), "count": self.process_count}
            )
            
            return ProcessResult.success_result(f"Mock processing completed (count: {self.process_count})")
        
        async def process_async(self, document: Document) -> ProcessResult:
            # Simulate async processing delay
            await asyncio.sleep(0.01)
            return self.process(document)
        
        def supports_file_type(self, file_type: str) -> bool:
            return True  # Mock processor supports all file types
    
    return MockProcessor


@pytest.fixture
def basic_pipeline(mock_processor):
    """Create a basic pipeline for testing."""
    pipeline = Pipeline("TestPipeline")
    pipeline.add_processor(mock_processor())
    return pipeline


@pytest.fixture
def multi_processor_pipeline(mock_processor):
    """Create a pipeline with multiple processors."""
    pipeline = Pipeline("MultiProcessorPipeline")
    
    # Add multiple processors
    pipeline.add_processor(mock_processor(name="Processor1"))
    pipeline.add_processor(mock_processor(name="Processor2"))
    pipeline.add_processor(mock_processor(name="Processor3"))
    
    return pipeline


@pytest.fixture
def chat_session():
    """Create a chat session for testing."""
    session = ChatSession(
        session_id="test-session-123",
        metadata={"test": True, "created_by": "pytest"}
    )
    
    # Add some sample messages
    session.add_message("user", "Hello, this is a test message", metadata={"test": True})
    session.add_message("assistant", "Hello! How can I help you?", metadata={"test": True})
    
    return session


@pytest.fixture
def session_manager():
    """Create a session manager for testing."""
    manager = ChatSessionManager()
    
    # Add some test sessions
    for i in range(3):
        session_id = f"test-session-{i}"
        session = manager.create_session(
            session_id=session_id,
            metadata={"test": True, "session_index": i}
        )
        session.add_message("user", f"Test message {i}")
    
    return manager


@pytest.fixture
def mock_api_responses():
    """Mock API responses for external services."""
    return {
        "ocr_response": {
            "status": "success",
            "text": "This is extracted text from OCR processing",
            "confidence": 0.95,
            "regions": [
                {"text": "This is extracted text", "bbox": [10, 10, 200, 30]},
                {"text": "from OCR processing", "bbox": [10, 35, 180, 55]}
            ]
        },
        "llm_response": {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "This is a mock LLM response for testing purposes."
                }
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70
            }
        },
        "embedding_response": {
            "data": [{
                "embedding": [0.1] * 128,  # Mock 128-dimensional embedding
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        }
    }


@pytest.fixture
def mock_external_apis(mock_api_responses):
    """Mock external API calls."""
    with patch('requests.post') as mock_post, \
         patch('aiohttp.ClientSession.post') as mock_aiohttp_post:
        
        # Configure synchronous requests mock
        mock_post.return_value.json.return_value = mock_api_responses["llm_response"]
        mock_post.return_value.status_code = 200
        
        # Configure async requests mock
        mock_response = MagicMock()
        mock_response.json = asyncio.coroutine(lambda: mock_api_responses["llm_response"])
        mock_response.status = 200
        mock_aiohttp_post.return_value.__aenter__.return_value = mock_response
        
        yield {
            'requests_post': mock_post,
            'aiohttp_post': mock_aiohttp_post,
            'responses': mock_api_responses
        }


@pytest.fixture
def performance_monitor():
    """Create a performance monitoring fixture."""
    memory_manager = MemoryManager()
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_memory = None
            self.start_time = None
            self.metrics = []
        
        def start_monitoring(self, operation_name="test_operation"):
            self.operation_name = operation_name
            self.start_memory = memory_manager.get_memory_usage()
            self.start_time = asyncio.get_event_loop().time()
        
        def stop_monitoring(self):
            if self.start_time is None:
                return None
            
            end_time = asyncio.get_event_loop().time()
            end_memory = memory_manager.get_memory_usage()
            
            metrics = {
                'operation': self.operation_name,
                'duration': end_time - self.start_time,
                'memory_before': self.start_memory['used'],
                'memory_after': end_memory['used'],
                'memory_increase': end_memory['used'] - self.start_memory['used'],
                'memory_percent': end_memory['percent']
            }
            
            self.metrics.append(metrics)
            return metrics
        
        def assert_performance(self, max_duration=None, max_memory_increase=None):
            if not self.metrics:
                pytest.fail("No performance metrics available. Call start_monitoring() and stop_monitoring() first.")
            
            latest = self.metrics[-1]
            
            if max_duration and latest['duration'] > max_duration:
                pytest.fail(f"Operation took {latest['duration']:.3f}s, exceeding maximum of {max_duration}s")
            
            if max_memory_increase and latest['memory_increase'] > max_memory_increase:
                pytest.fail(f"Memory increased by {latest['memory_increase']:.2f}GB, exceeding maximum of {max_memory_increase}GB")
    
    return PerformanceMonitor()


@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatically cleanup test environment after each test."""
    yield
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear any global state
    # This can be extended based on specific needs


@pytest.fixture
def async_test_client():
    """Create an async test client for web application testing."""
    from webapp import app
    
    class AsyncTestClient:
        def __init__(self, app):
            self.app = app
        
        async def post(self, path, data=None, json=None, **kwargs):
            with self.app.test_client() as client:
                if json:
                    kwargs['json'] = json
                if data:
                    kwargs['data'] = data
                return client.post(path, **kwargs)
        
        async def get(self, path, **kwargs):
            with self.app.test_client() as client:
                return client.get(path, **kwargs)
    
    return AsyncTestClient(app)


@pytest.fixture
def test_data_generator():
    """Generate test data on demand."""
    class TestDataGenerator:
        @staticmethod
        def create_document_data(count=1, file_type="pdf"):
            """Generate document data for testing."""
            documents = []
            for i in range(count):
                doc_data = {
                    "document_id": f"test-doc-{i:03d}",
                    "file_name": f"test_document_{i}.{file_type}",
                    "file_type": file_type,
                    "content": {
                        "raw_text": f"This is test document number {i}. " * 10,
                        "metadata": {"page_count": i + 1, "language": "en"}
                    },
                    "tags": [f"test-{i}", "generated"],
                    "creation_time": datetime.now().isoformat()
                }
                documents.append(doc_data)
            return documents if count > 1 else documents[0]
        
        @staticmethod
        def create_chat_messages(count=1):
            """Generate chat messages for testing."""
            messages = []
            for i in range(count):
                user_msg = {
                    "role": "user",
                    "content": f"This is test message {i}",
                    "timestamp": datetime.now().isoformat()
                }
                assistant_msg = {
                    "role": "assistant", 
                    "content": f"This is response {i}",
                    "timestamp": datetime.now().isoformat()
                }
                messages.extend([user_msg, assistant_msg])
            return messages
        
        @staticmethod
        def create_embeddings(dimensions=128, count=1):
            """Generate mock embeddings."""
            import random
            embeddings = []
            for _ in range(count):
                embedding = [random.uniform(-1, 1) for _ in range(dimensions)]
                embeddings.append(embedding)
            return embeddings if count > 1 else embeddings[0]
    
    return TestDataGenerator()


# Performance assertion helpers
def assert_memory_usage_reasonable(operation_name="test", max_increase_gb=0.5):
    """Assert that memory usage increase is reasonable."""
    def decorator(func):
        @pytest.mark.asyncio
        async def wrapper(*args, **kwargs):
            memory_before = MemoryManager.get_memory_usage()
            
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            memory_after = MemoryManager.get_memory_usage()
            memory_increase = memory_after['used'] - memory_before['used']
            
            assert memory_increase < max_increase_gb, \
                f"Memory usage increased by {memory_increase:.2f}GB for {operation_name}, " \
                f"exceeding limit of {max_increase_gb}GB"
            
            return result
        return wrapper
    return decorator


def assert_execution_time(max_seconds=10.0):
    """Assert that execution time is within acceptable limits."""
    def decorator(func):
        @pytest.mark.asyncio
        async def wrapper(*args, **kwargs):
            start_time = asyncio.get_event_loop().time()
            
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            duration = asyncio.get_event_loop().time() - start_time
            
            assert duration < max_seconds, \
                f"Execution took {duration:.3f}s, exceeding limit of {max_seconds}s"
            
            return result
        return wrapper
    return decorator


# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable verbose logging for external libraries during tests
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)