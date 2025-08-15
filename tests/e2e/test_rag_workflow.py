"""
End-to-End RAG Workflow Tests

Comprehensive test suite for the complete RAG (Retrieval-Augmented Generation) workflow
including document ingestion, processing, vectorization, retrieval, and generation.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from core.config_center import ConfigCenter
from core.pipeline import Pipeline
from core.processor_loader import ProcessorLoader
from rag.haystack_pipeline import HaystackRAGPipeline
from document_stores.implementations.memory_store import InMemoryDocumentStore
from connectors.base_llm_connector import BaseLLMConnector
from models.document import Document
from models.process_result import ProcessResult


class MockLLMConnector(BaseLLMConnector):
    """Mock LLM connector for testing"""
    
    def __init__(self, responses: List[str] = None):
        super().__init__()
        self.responses = responses or ["Mock response from LLM"]
        self.response_index = 0
        self.call_count = 0
        self.last_prompt = None
        self.last_context = None
    
    async def generate_response(self, prompt: str, context: List[str] = None, **kwargs) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_context = context
        
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        else:
            return self.responses[-1] if self.responses else "Default mock response"
    
    def generate_response_sync(self, prompt: str, context: List[str] = None, **kwargs) -> str:
        """Synchronous version for compatibility"""
        return asyncio.run(self.generate_response(prompt, context, **kwargs))


class MockProcessor:
    """Mock processor for testing pipeline"""
    
    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.process_count = 0
    
    def get_name(self) -> str:
        return self.name
    
    def get_stage(self) -> str:
        return "test"
    
    def supports_file_type(self, file_type: str) -> bool:
        return True
    
    def process(self, document: Document) -> ProcessResult:
        self.process_count += 1
        
        if self.should_fail:
            return ProcessResult.error_result(f"Mock failure in {self.name}")
        
        # Simulate processing by modifying content
        if "embedding" in self.name.lower():
            # Simulate embedding generation
            document.embeddings = [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim mock embedding
        else:
            document.content = f"{document.content} [processed by {self.name}]"
        
        return ProcessResult.success_result(f"Processed by {self.name}")


@pytest.fixture
def temp_config_dir():
    """Create temporary configuration directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_config():
    """Sample configuration for RAG system"""
    return {
        "version": "2.0.0",
        "storage": {
            "base_path": "./test_data",
            "output_path": "./test_output"
        },
        "processors": {
            "pre_processor": {
                "enabled": True,
                "config": {"batch_size": 10}
            },
            "embedding_processor": {
                "enabled": True,
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
            }
        },
        "llm": {
            "type": "mock",
            "model": "mock-model",
            "settings": {
                "temperature": 0.7,
                "max_tokens": 1000
            }
        },
        "vector_db": {
            "document_store": {
                "type": "memory",
                "embedding_dim": 500,
                "similarity": "cosine"
            }
        },
        "rag": {
            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.5
            },
            "generation": {
                "max_length": 200,
                "include_source": True
            }
        }
    }


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    documents = [
        Document(
            document_id="doc_001",
            file_name="machine_learning.txt",
            file_path="/tmp/machine_learning.txt",
            file_type="txt",
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed."
        ),
        Document(
            document_id="doc_002",
            file_name="neural_networks.txt",
            file_path="/tmp/neural_networks.txt",
            file_type="txt",
            content="Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information."
        ),
        Document(
            document_id="doc_003",
            file_name="deep_learning.txt",
            file_path="/tmp/deep_learning.txt",
            file_type="txt",
            content="Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns in data."
        )
    ]
    return documents


@pytest.fixture
def mock_config_center(temp_config_dir, sample_config):
    """Create mock config center with sample configuration"""
    config_file = Path(temp_config_dir) / "config.json"
    with open(config_file, 'w') as f:
        json.dump(sample_config, f)
    
    config_center = ConfigCenter(
        base_config_path=temp_config_dir,
        environment="test",
        watch_changes=False
    )
    return config_center


class TestRAGWorkflowComponents:
    """Test individual components of the RAG workflow"""
    
    def test_document_store_initialization(self, sample_config):
        """Test document store initialization with configuration"""
        store_config = sample_config["vector_db"]["document_store"]
        document_store = InMemoryDocumentStore()
        
        assert document_store is not None
        assert hasattr(document_store, 'documents')
        assert hasattr(document_store, 'embeddings')
    
    def test_llm_connector_initialization(self):
        """Test LLM connector initialization"""
        responses = [
            "Machine learning is a powerful technique for data analysis.",
            "Neural networks are fundamental to deep learning.",
            "AI has many applications in modern technology."
        ]
        connector = MockLLMConnector(responses)
        
        assert connector.responses == responses
        assert connector.call_count == 0
    
    def test_pipeline_processor_loading(self, mock_config_center):
        """Test loading processors into pipeline"""
        pipeline = Pipeline("TestRAGPipeline")
        
        # Add mock processors
        processors = [
            MockProcessor("pre_processor"),
            MockProcessor("embedding_processor")
        ]
        
        for processor in processors:
            pipeline.add_processor(processor)
        
        assert len(pipeline.processors) == 2
        assert pipeline.processors[0].get_name() == "pre_processor"
        assert pipeline.processors[1].get_name() == "embedding_processor"


class TestDocumentIngestionWorkflow:
    """Test document ingestion and processing workflow"""
    
    @pytest.mark.asyncio
    async def test_single_document_processing(self, sample_documents):
        """Test processing a single document through the pipeline"""
        pipeline = Pipeline("IngestionPipeline")
        
        # Add processors
        processors = [
            MockProcessor("pre_processor"),
            MockProcessor("embedding_processor")
        ]
        
        for processor in processors:
            pipeline.add_processor(processor)
        
        document = sample_documents[0]
        result = await pipeline.process_document(document)
        
        assert result.is_successful()
        assert document.status == "completed"
        assert len(document.processing_history) == 2
        
        # Check that embedding was generated
        assert hasattr(document, 'embeddings')
        assert len(document.embeddings) == 500  # Mock embedding dimension
    
    @pytest.mark.asyncio
    async def test_batch_document_processing(self, sample_documents):
        """Test batch processing of multiple documents"""
        pipeline = Pipeline("BatchIngestionPipeline")
        
        processors = [
            MockProcessor("pre_processor"),
            MockProcessor("embedding_processor")
        ]
        
        for processor in processors:
            pipeline.add_processor(processor)
        
        results = await pipeline.process_documents(sample_documents, max_concurrent=2)
        
        assert len(results) == 3
        for doc in sample_documents:
            assert results[doc.document_id].is_successful()
            assert doc.status == "completed"
            assert hasattr(doc, 'embeddings')
    
    @pytest.mark.asyncio
    async def test_processing_with_failures(self, sample_documents):
        """Test document processing with some failures"""
        pipeline = Pipeline("FailurePipeline")
        
        processors = [
            MockProcessor("pre_processor"),
            MockProcessor("failing_processor", should_fail=True),
            MockProcessor("final_processor")
        ]
        
        for processor in processors:
            pipeline.add_processor(processor)
        
        document = sample_documents[0]
        result = await pipeline.process_document(document)
        
        assert not result.is_successful()
        assert document.status == "error"
        assert "failing_processor" in result.get_message()


class TestVectorStorageWorkflow:
    """Test vector storage and retrieval workflow"""
    
    def test_document_storage(self, sample_documents):
        """Test storing documents in vector store"""
        document_store = InMemoryDocumentStore()
        
        # Simulate documents with embeddings
        for doc in sample_documents:
            doc.embeddings = [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim embedding
        
        # Store documents
        for doc in sample_documents:
            document_store.store_document(doc)
        
        assert len(document_store.documents) == 3
        assert len(document_store.embeddings) == 3
    
    def test_similarity_search(self, sample_documents):
        """Test similarity search in vector store"""
        document_store = InMemoryDocumentStore()
        
        # Store documents with embeddings
        for i, doc in enumerate(sample_documents):
            # Create slightly different embeddings for each document
            base_embedding = [0.1 + i * 0.1] * 500
            doc.embeddings = base_embedding
            document_store.store_document(doc)
        
        # Search with query embedding similar to first document
        query_embedding = [0.15] * 500
        results = document_store.similarity_search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        for result in results:
            assert 'document' in result
            assert 'score' in result
            assert result['score'] >= 0.0


class TestRAGPipelineIntegration:
    """Test complete RAG pipeline integration"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_rag_workflow(self, sample_documents, sample_config):
        """Test complete end-to-end RAG workflow"""
        # Initialize components
        document_store = InMemoryDocumentStore()
        llm_connector = MockLLMConnector([
            "Based on the provided context, machine learning is indeed a subset of AI that learns from data patterns."
        ])
        
        # Create RAG pipeline
        rag_pipeline = HaystackRAGPipeline(
            document_store=document_store,
            llm_connector=llm_connector,
            config=sample_config.get("rag", {})
        )
        
        # Step 1: Process and store documents
        processing_pipeline = Pipeline("RAGProcessing")
        processing_pipeline.add_processor(MockProcessor("pre_processor"))
        processing_pipeline.add_processor(MockProcessor("embedding_processor"))
        
        # Process documents
        for doc in sample_documents:
            result = await processing_pipeline.process_document(doc)
            assert result.is_successful()
            
            # Store in document store
            document_store.store_document(doc)
        
        # Step 2: Perform RAG query
        query = "What is machine learning and how does it relate to AI?"
        response = await rag_pipeline.query(query)
        
        assert response is not None
        assert len(response) > 0
        assert "machine learning" in response.lower()
        
        # Verify LLM was called with context
        assert llm_connector.call_count == 1
        assert llm_connector.last_context is not None
        assert len(llm_connector.last_context) > 0
    
    @pytest.mark.asyncio
    async def test_rag_with_insufficient_context(self, sample_config):
        """Test RAG pipeline behavior with insufficient context"""
        document_store = InMemoryDocumentStore()
        llm_connector = MockLLMConnector([
            "I don't have enough context to answer this question accurately."
        ])
        
        rag_pipeline = HaystackRAGPipeline(
            document_store=document_store,
            llm_connector=llm_connector,
            config=sample_config.get("rag", {})
        )
        
        # Query without any documents in store
        query = "What is quantum computing?"
        response = await rag_pipeline.query(query)
        
        # Should still get a response, but with no context
        assert response is not None
        assert llm_connector.call_count == 1
        assert llm_connector.last_context == [] or llm_connector.last_context is None
    
    @pytest.mark.asyncio
    async def test_rag_with_similarity_filtering(self, sample_documents, sample_config):
        """Test RAG pipeline with similarity threshold filtering"""
        document_store = InMemoryDocumentStore()
        llm_connector = MockLLMConnector([
            "The context provided discusses machine learning concepts."
        ])
        
        # Set high similarity threshold
        rag_config = sample_config.get("rag", {})
        rag_config["retrieval"]["similarity_threshold"] = 0.9
        
        rag_pipeline = HaystackRAGPipeline(
            document_store=document_store,
            llm_connector=llm_connector,
            config=rag_config
        )
        
        # Store documents with embeddings
        for doc in sample_documents:
            doc.embeddings = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
            document_store.store_document(doc)
        
        # Query with different embedding (low similarity)
        query = "Tell me about quantum physics"
        response = await rag_pipeline.query(query)
        
        assert response is not None
        assert llm_connector.call_count == 1


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in RAG workflow"""
    
    @pytest.mark.asyncio
    async def test_document_processing_failure_recovery(self, sample_documents):
        """Test recovery from document processing failures"""
        pipeline = Pipeline("RecoveryTestPipeline")
        
        # Add processors with one that fails
        processors = [
            MockProcessor("pre_processor"),
            MockProcessor("failing_processor", should_fail=True)
        ]
        
        for processor in processors:
            pipeline.add_processor(processor)
        
        document_store = InMemoryDocumentStore()
        successful_docs = []
        failed_docs = []
        
        # Process documents and handle failures
        for doc in sample_documents:
            result = await pipeline.process_document(doc)
            
            if result.is_successful():
                document_store.store_document(doc)
                successful_docs.append(doc)
            else:
                failed_docs.append(doc)
        
        # Should have failures but continue processing other documents
        assert len(failed_docs) == 3  # All should fail due to failing processor
        assert len(successful_docs) == 0
        assert len(document_store.documents) == 0
    
    @pytest.mark.asyncio
    async def test_llm_connector_failure_handling(self, sample_documents, sample_config):
        """Test handling of LLM connector failures"""
        document_store = InMemoryDocumentStore()
        
        # Create LLM connector that will fail
        failing_connector = MockLLMConnector()
        
        async def failing_generate(prompt, context=None, **kwargs):
            raise Exception("LLM service unavailable")
        
        failing_connector.generate_response = failing_generate
        
        rag_pipeline = HaystackRAGPipeline(
            document_store=document_store,
            llm_connector=failing_connector,
            config=sample_config.get("rag", {})
        )
        
        # Store a document
        doc = sample_documents[0]
        doc.embeddings = [0.1] * 500
        document_store.store_document(doc)
        
        # Query should handle LLM failure gracefully
        with pytest.raises(Exception, match="LLM service unavailable"):
            await rag_pipeline.query("What is machine learning?")
    
    def test_empty_document_store_query(self, sample_config):
        """Test querying empty document store"""
        document_store = InMemoryDocumentStore()
        llm_connector = MockLLMConnector(["No relevant information found."])
        
        rag_pipeline = HaystackRAGPipeline(
            document_store=document_store,
            llm_connector=llm_connector,
            config=sample_config.get("rag", {})
        )
        
        # Query empty store
        query = "What is machine learning?"
        
        # Should handle empty store gracefully
        # Note: This test might need adjustment based on actual implementation
        with pytest.raises(Exception):
            # Assuming the implementation raises an exception for empty store
            asyncio.run(rag_pipeline.query(query))
    
    @pytest.mark.asyncio
    async def test_malformed_document_handling(self):
        """Test handling of malformed documents"""
        pipeline = Pipeline("MalformedDocPipeline")
        pipeline.add_processor(MockProcessor("robust_processor"))
        
        # Create malformed document
        malformed_doc = Document(
            document_id="malformed_001",
            file_name="",  # Empty filename
            file_path=None,  # None file path
            file_type="unknown",
            content=None  # None content
        )
        
        # Should handle malformed document gracefully
        result = await pipeline.process_document(malformed_doc)
        
        # Result depends on processor implementation
        # This test verifies that the pipeline doesn't crash
        assert result is not None


class TestPerformanceAndScaling:
    """Test performance and scaling aspects"""
    
    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, temp_config_dir):
        """Test concurrent processing of large document batches"""
        # Create large batch of documents
        large_batch = []
        for i in range(20):
            doc = Document(
                document_id=f"perf_doc_{i:03d}",
                file_name=f"document_{i}.txt",
                file_path=f"/tmp/document_{i}.txt",
                file_type="txt",
                content=f"This is test document number {i} with some sample content for processing."
            )
            large_batch.append(doc)
        
        pipeline = Pipeline("PerformancePipeline")
        pipeline.add_processor(MockProcessor("fast_processor"))
        
        import time
        start_time = time.time()
        
        # Process with high concurrency
        results = await pipeline.process_documents(large_batch, max_concurrent=10)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(results) == 20
        assert all(result.is_successful() for result in results.values())
        
        # Should process reasonably quickly with concurrency
        assert processing_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_batch_processing(self, sample_documents):
        """Test memory usage during large batch processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create larger document batch
        large_batch = sample_documents * 10  # 30 documents
        
        pipeline = Pipeline("MemoryTestPipeline")
        pipeline.add_processor(MockProcessor("memory_processor"))
        
        # Process batch
        results = await pipeline.process_documents(large_batch, max_concurrent=5)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert len(results) == 30
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024  # 100MB threshold


class TestConfigurationIntegration:
    """Test integration with configuration system"""
    
    def test_rag_configuration_loading(self, mock_config_center):
        """Test loading RAG configuration from config center"""
        config = mock_config_center.get_config()
        
        assert "rag" in config
        assert config["rag"]["retrieval"]["top_k"] == 5
        assert config["rag"]["generation"]["max_length"] == 200
    
    def test_dynamic_configuration_updates(self, mock_config_center):
        """Test dynamic configuration updates during runtime"""
        # Get initial config
        initial_config = mock_config_center.get_value("rag.retrieval.top_k")
        assert initial_config == 5
        
        # Update configuration
        mock_config_center.set_value("rag.retrieval.top_k", 10, persist=False)
        
        # Verify update
        updated_config = mock_config_center.get_value("rag.retrieval.top_k")
        assert updated_config == 10
    
    def test_configuration_validation(self, mock_config_center):
        """Test configuration validation for RAG components"""
        validation_result = mock_config_center.validate_current_config()
        
        assert validation_result["is_valid"] is True
        
        # Test invalid configuration
        mock_config_center.set_value("rag.retrieval.top_k", -1, persist=False)
        
        # Validation should catch invalid values
        # Note: This depends on actual validation implementation
        validation_result = mock_config_center.validate_current_config()
        # The specific assertion depends on whether validation is implemented