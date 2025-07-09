"""
Document processing pipeline component test
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from core.pipeline import Pipeline
from models.document import Document
from core.process_result import ProcessResult
from processors.base_processor import IProcessor


class MockProcessor(IProcessor):
    """Mock processor for testing"""
    
    def __init__(self, name, success=True, side_effect=None):
        self.name = name
        self.success = success
        self.side_effect = side_effect
        self.called = False
        self.input_document = None
    
    def process(self, document):
        """Process the document"""
        self.called = True
        self.input_document = document
        
        if self.side_effect:
            self.side_effect(document)
        
        return ProcessResult(
            success=self.success,
            message=f"{self.name} processed",
            data={"processor_name": self.name}
        )


class TestProcessingPipeline:
    """Processing pipeline component test"""
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing"""
        doc = Document("test.pdf")
        doc.file_path = "test.pdf"
        doc.file_type = "pdf"
        return doc
    
    @pytest.fixture
    def basic_pipeline(self):
        """Create a basic pipeline with mock processors"""
        pipeline = Pipeline("TestPipeline")
        
        # Add processors
        pipeline.add_processor(MockProcessor("Preprocessor"))
        pipeline.add_processor(MockProcessor("OCRProcessor"))
        pipeline.add_processor(MockProcessor("ClassificationProcessor"))
        
        return pipeline
    
    def test_pipeline_execution(self, basic_pipeline, sample_document):
        """Test the execution of a processing pipeline"""
        # Process the document
        result = basic_pipeline.process_document(sample_document)
        
        # Verify the result
        assert result.is_successful()
        assert "Preprocessor" in result.data
        assert "OCRProcessor" in result.data
        assert "ClassificationProcessor" in result.data
        
        # Verify all processors were called
        for processor in basic_pipeline.processors:
            assert processor.called
            assert processor.input_document == sample_document
    
    def test_pipeline_with_failing_processor(self, sample_document):
        """Test pipeline behavior when a processor fails"""
        pipeline = Pipeline("FailingPipeline")
        
        # Add processors with the second one failing
        pipeline.add_processor(MockProcessor("FirstProcessor"))
        pipeline.add_processor(MockProcessor("FailingProcessor", success=False))
        pipeline.add_processor(MockProcessor("ThirdProcessor"))
        
        # Process the document
        result = pipeline.process_document(sample_document)
        
        # Verify the result indicates failure
        assert not result.is_successful()
        assert "FirstProcessor" in result.data
        assert "FailingProcessor" in result.data
        assert "ThirdProcessor" not in result.data
        
        # Verify only the first two processors were called
        assert pipeline.processors[0].called
        assert pipeline.processors[1].called
        assert not pipeline.processors[2].called
    
    def test_pipeline_with_document_modification(self, sample_document):
        """Test pipeline where processors modify the document"""
        def modify_document(doc):
            """Add content to the document"""
            doc.content = "OCR extracted text"
            doc.metadata["language"] = "en"
        
        def add_classification(doc):
            """Add classification to the document"""
            doc.metadata["category"] = "science"
        
        pipeline = Pipeline("ModificationPipeline")
        
        # Add processors that modify the document
        pipeline.add_processor(MockProcessor("ContentProcessor", side_effect=modify_document))
        pipeline.add_processor(MockProcessor("ClassificationProcessor", side_effect=add_classification))
        
        # Process the document
        result = pipeline.process_document(sample_document)
        
        # Verify the result
        assert result.is_successful()
        
        # Verify document modifications
        assert sample_document.content == "OCR extracted text"
        assert sample_document.metadata["language"] == "en"
        assert sample_document.metadata["category"] == "science"
    
    def test_pipeline_empty(self):
        """Test behavior of an empty pipeline"""
        pipeline = Pipeline("EmptyPipeline")
        document = Document("test.txt")
        
        result = pipeline.process_document(document)
        
        # An empty pipeline should succeed but do nothing
        assert result.is_successful()
        assert result.get_message() == "No processors in pipeline"
    
    def test_pipeline_execute_one_processor(self, sample_document):
        """Test executing just one processor in the pipeline"""
        pipeline = Pipeline("SelectivePipeline")
        
        processor1 = MockProcessor("Processor1")
        processor2 = MockProcessor("Processor2")
        processor3 = MockProcessor("Processor3")
        
        pipeline.add_processor(processor1)
        pipeline.add_processor(processor2)
        pipeline.add_processor(processor3)
        
        # Execute only the second processor
        result = pipeline.execute_processor(1, sample_document)
        
        # Verify only the second processor was called
        assert not processor1.called
        assert processor2.called
        assert not processor3.called
        
        # Verify the result
        assert result.is_successful()
        assert result.data["processor_name"] == "Processor2"
    
    def test_pipeline_with_real_temp_files(self):
        """Test pipeline with real temporary files"""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a test file
            test_file_path = os.path.join(temp_dir, "test.txt")
            with open(test_file_path, "w") as f:
                f.write("Test document content")
            
            # Create test output directory
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Define a processor that writes to the output directory
            def save_output(doc):
                """Save document to output directory"""
                output_path = os.path.join(output_dir, f"{doc.id}.txt")
                with open(output_path, "w") as f:
                    f.write(doc.content)
                doc.metadata["output_path"] = output_path
            
            pipeline = Pipeline("FileProcessingPipeline")
            pipeline.add_processor(MockProcessor("Reader"))
            pipeline.add_processor(MockProcessor("Writer", side_effect=save_output))
            
            # Create a document
            document = Document(test_file_path)
            document.file_path = test_file_path
            document.content = "Processed content"
            
            # Process the document
            result = pipeline.process_document(document)
            
            # Verify the result
            assert result.is_successful()
            
            # Verify the output file was created
            output_path = document.metadata.get("output_path")
            assert output_path is not None
            assert os.path.exists(output_path)
            
            # Verify the content
            with open(output_path, "r") as f:
                content = f.read()
                assert content == "Processed content"
                
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
