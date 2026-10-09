"""
Document processing pipeline component test
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from core.pipeline import Pipeline
from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor as IProcessor


class MockProcessor(IProcessor):
    """Mock processor for testing"""

    def __init__(self, name, success=True, side_effect=None):
        super().__init__(name=name)
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

        return ProcessResult(success=self.success, message=f"{self.name} processed", data={"processor_name": self.name})


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

    @pytest.mark.asyncio
    async def test_pipeline_execution(self, basic_pipeline, sample_document):
        """Test the execution of a processing pipeline"""
        # Process the document
        result = await basic_pipeline.process_document(sample_document)

        # Verify the result
        assert result.is_successful()
        assert [record["processor"] for record in sample_document.processing_history if "processor" in record] == [
            "Preprocessor",
            "OCRProcessor",
            "ClassificationProcessor",
        ]

        # Verify all processors were called
        for processor in basic_pipeline.processors:
            assert processor.called
            assert processor.input_document == sample_document

    @pytest.mark.asyncio
    async def test_pipeline_with_failing_processor(self, sample_document):
        """Test pipeline behavior when a processor fails"""
        pipeline = Pipeline("FailingPipeline")

        # Add processors with the second one failing
        pipeline.add_processor(MockProcessor("FirstProcessor"))
        pipeline.add_processor(MockProcessor("FailingProcessor", success=False))
        pipeline.add_processor(MockProcessor("ThirdProcessor"))

        # Process the document
        result = await pipeline.process_document(sample_document)

        # Verify the result indicates failure
        assert not result.is_successful()
        assert "FailingProcessor" in result.get_message()

        # Verify only the first two processors were called
        assert pipeline.processors[0].called
        assert pipeline.processors[1].called
        assert not pipeline.processors[2].called

    @pytest.mark.asyncio
    async def test_pipeline_with_document_modification(self, sample_document):
        """Test pipeline where processors modify the document"""

        def modify_document(doc):
            """Add content to the document"""
            doc.store_content("ContentProcessor", "Controlled fixture text")
            doc.metadata["language"] = "en"

        def add_classification(doc):
            """Add classification to the document"""
            doc.metadata["category"] = "science"

        pipeline = Pipeline("ModificationPipeline")

        # Add processors that modify the document
        pipeline.add_processor(MockProcessor("ContentProcessor", side_effect=modify_document))
        pipeline.add_processor(MockProcessor("ClassificationProcessor", side_effect=add_classification))

        # Process the document
        result = await pipeline.process_document(sample_document)

        # Verify the result
        assert result.is_successful()

        # Verify document modifications
        assert sample_document.get_content("ContentProcessor") == "Controlled fixture text"
        assert sample_document.metadata["language"] == "en"
        assert sample_document.metadata["category"] == "science"

    @pytest.mark.asyncio
    async def test_pipeline_empty(self):
        """Test behavior of an empty pipeline"""
        pipeline = Pipeline("EmptyPipeline")
        document = Document("test.txt")

        result = await pipeline.process_document(document)

        assert not result.is_successful()
        assert "没有处理器" in result.get_message()

    @pytest.mark.asyncio
    async def test_pipeline_start_from_processor(self, sample_document):
        """Test executing just one processor in the pipeline"""
        pipeline = Pipeline("SelectivePipeline")

        processor1 = MockProcessor("Processor1")
        processor2 = MockProcessor("Processor2")
        processor3 = MockProcessor("Processor3")

        pipeline.add_processor(processor1)
        pipeline.add_processor(processor2)
        pipeline.add_processor(processor3)

        # Execute only the second processor
        result = await pipeline.process_document(sample_document, start_from="Processor2")

        # Verify only the second processor was called
        assert not processor1.called
        assert processor2.called
        assert processor3.called

        # Verify the result
        assert result.is_successful()
        assert [record["processor"] for record in sample_document.processing_history if "processor" in record] == [
            "Processor2",
            "Processor3",
        ]

    @pytest.mark.asyncio
    async def test_pipeline_with_real_temp_files(self):
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
                output_path = os.path.join(output_dir, f"{doc.document_id}.txt")
                with open(output_path, "w") as f:
                    f.write(doc.get_content("fixture"))
                doc.metadata["output_path"] = output_path

            pipeline = Pipeline("FileProcessingPipeline")
            pipeline.add_processor(MockProcessor("Reader"))
            pipeline.add_processor(MockProcessor("Writer", side_effect=save_output))

            # Create a document
            document = Document(test_file_path)
            document.file_path = test_file_path
            document.store_content("fixture", "Processed content")

            # Process the document
            result = await pipeline.process_document(document)

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
