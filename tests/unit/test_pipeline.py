"""
Tests for core.pipeline module

Comprehensive test suite covering Pipeline class functionality including:
- Processor management (add, remove, reorder)
- Document processing (sync and async)
- Batch processing with concurrency control
- Error handling and edge cases
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from core.pipeline import Pipeline
from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import IProcessor


class MockProcessor(IProcessor):
    """Mock processor for testing purposes"""
    
    def __init__(self, name: str, stage: str = "test", supports_types: List[str] = None,
                 should_fail: bool = False, has_async: bool = False):
        self.name = name
        self.stage = stage
        self.supports_types = supports_types or ["txt", "pdf", "docx"]
        self.should_fail = should_fail
        self.has_async = has_async
        self.process_count = 0
        self.async_process_count = 0
    
    def get_name(self) -> str:
        return self.name
    
    def get_stage(self) -> str:
        return self.stage
    
    def supports_file_type(self, file_type: str) -> bool:
        return file_type in self.supports_types
    
    def process(self, document: Document) -> ProcessResult:
        self.process_count += 1
        if self.should_fail:
            return ProcessResult.error_result(f"Mock failure in {self.name}")
        
        # Modify document to simulate processing
        document.content = f"{document.content} -> processed by {self.name}"
        return ProcessResult.success_result(f"Processed by {self.name}")
    
    async def process_async(self, document: Document) -> ProcessResult:
        self.async_process_count += 1
        if self.should_fail:
            return ProcessResult.error_result(f"Mock async failure in {self.name}")
        
        # Simulate async processing delay
        await asyncio.sleep(0.01)
        
        # Modify document to simulate processing
        document.content = f"{document.content} -> async processed by {self.name}"
        return ProcessResult.success_result(f"Async processed by {self.name}")


@pytest.fixture
def sample_document():
    """Create a sample document for testing"""
    doc = Document(
        document_id="test_doc_001",
        file_name="test_document.txt",
        file_path="/tmp/test_document.txt",
        file_type="txt",
        content="Initial content"
    )
    return doc


@pytest.fixture
def sample_documents():
    """Create multiple sample documents for batch testing"""
    documents = []
    for i in range(5):
        doc = Document(
            document_id=f"test_doc_{i:03d}",
            file_name=f"test_document_{i}.txt",
            file_path=f"/tmp/test_document_{i}.txt",
            file_type="txt",
            content=f"Initial content {i}"
        )
        documents.append(doc)
    return documents


@pytest.fixture
def mock_processors():
    """Create mock processors for testing"""
    return [
        MockProcessor("preprocessor", "preprocessing"),
        MockProcessor("ocr", "ocr"),
        MockProcessor("analyzer", "analysis")
    ]


class TestPipelineInitialization:
    """Test pipeline initialization and basic setup"""
    
    def test_default_initialization(self):
        """Test pipeline initialization with default values"""
        pipeline = Pipeline()
        
        assert pipeline.name == "DefaultPipeline"
        assert pipeline.processors == []
        assert isinstance(pipeline.logger, logging.Logger)
    
    def test_custom_name_initialization(self):
        """Test pipeline initialization with custom name"""
        pipeline = Pipeline(name="CustomPipeline")
        
        assert pipeline.name == "CustomPipeline"
        assert pipeline.processors == []


class TestProcessorManagement:
    """Test processor management functionality"""
    
    def test_add_processor(self, mock_processors):
        """Test adding processors to pipeline"""
        pipeline = Pipeline()
        processor = mock_processors[0]
        
        pipeline.add_processor(processor)
        
        assert len(pipeline.processors) == 1
        assert pipeline.processors[0] == processor
    
    def test_add_multiple_processors(self, mock_processors):
        """Test adding multiple processors maintains order"""
        pipeline = Pipeline()
        
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        assert len(pipeline.processors) == 3
        assert [p.get_name() for p in pipeline.processors] == ["preprocessor", "ocr", "analyzer"]
    
    def test_remove_processor_success(self, mock_processors):
        """Test successful processor removal"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        result = pipeline.remove_processor("ocr")
        
        assert result is True
        assert len(pipeline.processors) == 2
        assert [p.get_name() for p in pipeline.processors] == ["preprocessor", "analyzer"]
    
    def test_remove_processor_not_found(self, mock_processors):
        """Test removing non-existent processor"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        result = pipeline.remove_processor("nonexistent")
        
        assert result is False
        assert len(pipeline.processors) == 3
    
    def test_get_processors(self, mock_processors):
        """Test getting processors returns a copy"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        processors = pipeline.get_processors()
        
        assert len(processors) == 3
        assert processors is not pipeline.processors  # Should be a copy
        assert processors == pipeline.processors
    
    def test_clear_processors(self, mock_processors):
        """Test clearing all processors"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        pipeline.clear_processors()
        
        assert len(pipeline.processors) == 0
    
    def test_reorder_processors_success(self, mock_processors):
        """Test successful processor reordering"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        result = pipeline.reorder_processors(["analyzer", "preprocessor", "ocr"])
        
        assert result is True
        assert [p.get_name() for p in pipeline.processors] == ["analyzer", "preprocessor", "ocr"]
    
    def test_reorder_processors_missing_name(self, mock_processors):
        """Test reordering with missing processor name"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        result = pipeline.reorder_processors(["analyzer", "nonexistent", "ocr"])
        
        assert result is False
        # Original order should be preserved
        assert [p.get_name() for p in pipeline.processors] == ["preprocessor", "ocr", "analyzer"]


class TestSynchronousProcessing:
    """Test synchronous document processing"""
    
    def test_process_document_sync_success(self, sample_document, mock_processors):
        """Test successful synchronous document processing"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        result = pipeline.process_document_sync(sample_document)
        
        assert result.is_successful()
        assert sample_document.status == "completed"
        assert len(sample_document.processing_history) == 3
        
        # Check each processor was called
        for i, processor in enumerate(mock_processors):
            assert processor.process_count == 1
            history = sample_document.processing_history[i]
            assert history["processor"] == processor.get_name()
            assert history["success"] is True
    
    def test_process_document_sync_empty_pipeline(self, sample_document):
        """Test processing with empty pipeline"""
        pipeline = Pipeline()
        
        result = pipeline.process_document_sync(sample_document)
        
        assert not result.is_successful()
        assert "流水线中没有处理器" in result.get_message()
    
    def test_process_document_sync_processor_failure(self, sample_document, mock_processors):
        """Test processing with processor failure"""
        pipeline = Pipeline()
        # Add failing processor
        failing_processor = MockProcessor("failing", should_fail=True)
        pipeline.add_processor(mock_processors[0])
        pipeline.add_processor(failing_processor)
        pipeline.add_processor(mock_processors[2])
        
        result = pipeline.process_document_sync(sample_document)
        
        assert not result.is_successful()
        assert "failing" in result.get_message()
        assert sample_document.status == "error"
        assert len(sample_document.processing_history) == 1  # Only first processor succeeded
    
    def test_process_document_sync_unsupported_file_type(self, sample_document, mock_processors):
        """Test processing with unsupported file type"""
        sample_document.file_type = "unsupported"
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        result = pipeline.process_document_sync(sample_document)
        
        # Should succeed but skip all processors
        assert result.is_successful()
        assert len(sample_document.processing_history) == 0
        for processor in mock_processors:
            assert processor.process_count == 0
    
    def test_process_document_sync_start_from(self, sample_document, mock_processors):
        """Test processing starting from specific processor"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        result = pipeline.process_document_sync(sample_document, start_from="ocr")
        
        assert result.is_successful()
        assert mock_processors[0].process_count == 0  # preprocessor skipped
        assert mock_processors[1].process_count == 1  # ocr processed
        assert mock_processors[2].process_count == 1  # analyzer processed
    
    def test_process_documents_sync_batch(self, sample_documents, mock_processors):
        """Test synchronous batch processing"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        results = pipeline.process_documents_sync(sample_documents)
        
        assert len(results) == 5
        for doc in sample_documents:
            assert doc.document_id in results
            assert results[doc.document_id].is_successful()
            assert doc.status == "completed"


class TestAsynchronousProcessing:
    """Test asynchronous document processing"""
    
    @pytest.mark.asyncio
    async def test_process_document_async_success(self, sample_document, mock_processors):
        """Test successful asynchronous document processing"""
        # Add async support to processors
        for processor in mock_processors:
            processor.has_async = True
        
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        result = await pipeline.process_document(sample_document)
        
        assert result.is_successful()
        assert sample_document.status == "completed"
        assert len(sample_document.processing_history) == 3
        
        # Check async methods were called
        for processor in mock_processors:
            assert processor.async_process_count == 1
            assert processor.process_count == 0  # Sync methods not called
    
    @pytest.mark.asyncio
    async def test_process_document_async_fallback_to_sync(self, sample_document, mock_processors):
        """Test async processing falls back to sync when async not available"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        result = await pipeline.process_document(sample_document)
        
        assert result.is_successful()
        assert sample_document.status == "completed"
        
        # Check sync methods were called (fallback)
        for processor in mock_processors:
            assert processor.process_count == 1
            assert processor.async_process_count == 0
    
    @pytest.mark.asyncio
    async def test_process_document_async_empty_pipeline(self, sample_document):
        """Test async processing with empty pipeline"""
        pipeline = Pipeline()
        
        result = await pipeline.process_document(sample_document)
        
        assert not result.is_successful()
        assert "流水线中没有处理器" in result.get_message()
    
    @pytest.mark.asyncio
    async def test_process_document_async_processor_failure(self, sample_document, mock_processors):
        """Test async processing with processor failure"""
        pipeline = Pipeline()
        # Add failing processor
        failing_processor = MockProcessor("failing", should_fail=True, has_async=True)
        pipeline.add_processor(mock_processors[0])
        pipeline.add_processor(failing_processor)
        pipeline.add_processor(mock_processors[2])
        
        result = await pipeline.process_document(sample_document)
        
        assert not result.is_successful()
        assert "failing" in result.get_message()
        assert sample_document.status == "error"
    
    @pytest.mark.asyncio
    async def test_process_documents_async_batch_success(self, sample_documents, mock_processors):
        """Test successful async batch processing"""
        pipeline = Pipeline()
        for processor in mock_processors:
            processor.has_async = True
            pipeline.add_processor(processor)
        
        results = await pipeline.process_documents(sample_documents, max_concurrent=3)
        
        assert len(results) == 5
        for doc in sample_documents:
            assert doc.document_id in results
            assert results[doc.document_id].is_successful()
            assert doc.status == "completed"
    
    @pytest.mark.asyncio
    async def test_process_documents_async_batch_with_failures(self, sample_documents, mock_processors):
        """Test async batch processing with some failures"""
        pipeline = Pipeline()
        # Make some processors fail based on document content
        for processor in mock_processors:
            processor.has_async = True
            pipeline.add_processor(processor)
        
        # Make one processor fail for specific documents
        failing_processor = MockProcessor("conditional_fail", has_async=True)
        
        async def selective_fail(document):
            if "2" in document.document_id:  # Fail for document 2
                return ProcessResult.error_result("Selective failure")
            return ProcessResult.success_result("Success")
        
        failing_processor.process_async = selective_fail
        pipeline.add_processor(failing_processor)
        
        results = await pipeline.process_documents(sample_documents, max_concurrent=2)
        
        assert len(results) == 5
        
        # Check that some succeeded and some failed
        success_count = sum(1 for r in results.values() if r.is_successful())
        failure_count = sum(1 for r in results.values() if not r.is_successful())
        
        assert success_count == 4  # 4 should succeed
        assert failure_count == 1   # 1 should fail
    
    @pytest.mark.asyncio
    async def test_process_documents_async_concurrency_control(self, sample_documents, mock_processors):
        """Test async batch processing respects concurrency limits"""
        pipeline = Pipeline()
        for processor in mock_processors:
            processor.has_async = True
            pipeline.add_processor(processor)
        
        # Use very low concurrency to test semaphore
        start_time = asyncio.get_event_loop().time()
        results = await pipeline.process_documents(sample_documents, max_concurrent=1)
        end_time = asyncio.get_event_loop().time()
        
        # With max_concurrent=1, processing should be sequential
        # Each document takes at least 0.01s * 3 processors = 0.03s minimum
        # 5 documents = at least 0.15s total
        duration = end_time - start_time
        assert duration >= 0.1  # Allow some tolerance
        
        assert len(results) == 5
        for result in results.values():
            assert result.is_successful()


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_processor_exception_handling(self, sample_document):
        """Test handling of processor exceptions"""
        pipeline = Pipeline()
        
        # Create processor that raises exception
        exception_processor = Mock(spec=IProcessor)
        exception_processor.get_name.return_value = "exception_processor"
        exception_processor.supports_file_type.return_value = True
        exception_processor.process.side_effect = RuntimeError("Test exception")
        
        pipeline.add_processor(exception_processor)
        
        result = pipeline.process_document_sync(sample_document)
        
        assert not result.is_successful()
        assert "Test exception" in result.get_message()
        assert sample_document.status == "error"
    
    @pytest.mark.asyncio
    async def test_async_processor_exception_handling(self, sample_document):
        """Test handling of async processor exceptions"""
        pipeline = Pipeline()
        
        # Create async processor that raises exception
        exception_processor = Mock(spec=IProcessor)
        exception_processor.get_name.return_value = "async_exception_processor"
        exception_processor.supports_file_type.return_value = True
        exception_processor.process_async = AsyncMock(side_effect=RuntimeError("Async test exception"))
        
        pipeline.add_processor(exception_processor)
        
        result = await pipeline.process_document(sample_document)
        
        assert not result.is_successful()
        assert "Async test exception" in result.get_message()
        assert sample_document.status == "error"
    
    @pytest.mark.asyncio
    async def test_async_batch_with_task_exceptions(self, sample_documents):
        """Test async batch processing with task exceptions"""
        pipeline = Pipeline()
        
        # Create processor that fails for certain documents
        selective_processor = Mock(spec=IProcessor)
        selective_processor.get_name.return_value = "selective_processor"
        selective_processor.supports_file_type.return_value = True
        
        async def selective_failure(document):
            if "002" in document.document_id:
                raise RuntimeError("Selective async exception")
            return ProcessResult.success_result("Success")
        
        selective_processor.process_async = selective_failure
        pipeline.add_processor(selective_processor)
        
        results = await pipeline.process_documents(sample_documents)
        
        # Should have results for all documents, including error results
        assert len(results) >= len(sample_documents)
        
        # Check that some results are error results from exceptions
        error_results = [r for r in results.values() if not r.is_successful()]
        assert len(error_results) > 0


class TestPerformanceAndThreading:
    """Test performance and threading aspects"""
    
    @pytest.mark.asyncio
    async def test_thread_pool_executor_creation(self, sample_document):
        """Test that thread pool executor is created correctly"""
        pipeline = Pipeline()
        
        # Add sync processor (will use thread pool)
        sync_processor = MockProcessor("sync_processor")
        pipeline.add_processor(sync_processor)
        
        # Process document - should create thread pool executor
        result = await pipeline.process_document(sample_document)
        
        assert result.is_successful()
        assert hasattr(pipeline, '_executor')
        assert pipeline._executor.thread_name_prefix == 'pipeline-worker'
    
    def test_logging_configuration(self):
        """Test that logging is configured correctly"""
        pipeline = Pipeline("TestPipeline")
        
        assert isinstance(pipeline.logger, logging.Logger)
        assert pipeline.logger.name == "pipeline.TestPipeline"
    
    @pytest.mark.asyncio
    async def test_processing_status_updates(self, sample_document, mock_processors):
        """Test that document status is updated correctly during processing"""
        pipeline = Pipeline()
        for processor in mock_processors:
            pipeline.add_processor(processor)
        
        # Document should start with default status
        initial_status = sample_document.status
        
        result = await pipeline.process_document(sample_document)
        
        assert result.is_successful()
        assert sample_document.status == "completed"
        
        # Check processing history
        assert len(sample_document.processing_history) == 3
        for i, processor in enumerate(mock_processors):
            history = sample_document.processing_history[i]
            assert history["processor"] == processor.get_name()
            assert history["stage"] == processor.get_stage()
            assert history["success"] is True


class TestIntegration:
    """Integration tests combining multiple features"""
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_processors(self, sample_document):
        """Test pipeline with mix of sync and async processors"""
        pipeline = Pipeline()
        
        # Add mix of sync and async processors
        sync_processor = MockProcessor("sync", has_async=False)
        async_processor = MockProcessor("async", has_async=True)
        another_sync = MockProcessor("sync2", has_async=False)
        
        pipeline.add_processor(sync_processor)
        pipeline.add_processor(async_processor)
        pipeline.add_processor(another_sync)
        
        result = await pipeline.process_document(sample_document)
        
        assert result.is_successful()
        
        # Check that appropriate methods were called
        assert sync_processor.process_count == 1  # Sync fallback
        assert sync_processor.async_process_count == 0
        
        assert async_processor.process_count == 0
        assert async_processor.async_process_count == 1  # Async method
        
        assert another_sync.process_count == 1  # Sync fallback
        assert another_sync.async_process_count == 0
    
    @pytest.mark.asyncio
    async def test_complex_pipeline_workflow(self, sample_documents):
        """Test complex pipeline workflow with multiple processors and batch processing"""
        pipeline = Pipeline("ComplexWorkflow")
        
        # Create processors with different characteristics
        processors = [
            MockProcessor("preprocessor", "preprocessing", has_async=True),
            MockProcessor("validator", "validation", supports_types=["txt", "pdf"]),
            MockProcessor("analyzer", "analysis", has_async=True),
            MockProcessor("formatter", "formatting", supports_types=["txt"])
        ]
        
        for processor in processors:
            pipeline.add_processor(processor)
        
        # Process batch with concurrency
        results = await pipeline.process_documents(sample_documents, max_concurrent=2)
        
        assert len(results) == len(sample_documents)
        
        # All txt documents should succeed
        for doc in sample_documents:
            if doc.file_type == "txt":
                assert results[doc.document_id].is_successful()
                assert doc.status == "completed"
                
                # Check processing history for each processor that supports txt
                expected_processors = ["preprocessor", "validator", "analyzer", "formatter"]
                assert len(doc.processing_history) == len(expected_processors)
                
                for i, expected_name in enumerate(expected_processors):
                    assert doc.processing_history[i]["processor"] == expected_name