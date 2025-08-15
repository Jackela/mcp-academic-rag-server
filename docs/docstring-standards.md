# Documentation Standards and Docstring Guidelines

Comprehensive documentation standards for the MCP Academic RAG Server ensuring consistent, maintainable, and professional code documentation.

## Overview

This document establishes the documentation standards for the MCP Academic RAG Server project, including:

- **Docstring Conventions**: Standardized docstring formats
- **Code Documentation**: Inline comments and documentation practices
- **API Documentation**: Automated documentation generation
- **Type Annotations**: Comprehensive type hinting standards
- **Examples and Usage**: Practical examples for complex functionality

## Docstring Standards

### Google Style Docstrings

We use Google-style docstrings with Sphinx extensions for consistency and automatic documentation generation.

#### Function/Method Docstrings

```python
def process_document(file_path: str, collection_id: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> DocumentResult:
    """Process a document and add it to the specified collection.
    
    This function handles document ingestion, content extraction, embedding generation,
    and storage in the vector database. Supports multiple file formats including PDF,
    DOCX, TXT, and Markdown.
    
    Args:
        file_path (str): Absolute path to the document file to process.
        collection_id (str): Unique identifier for the target document collection.
        metadata (Optional[Dict[str, Any]]): Additional metadata to associate with
            the document. Common fields include 'author', 'title', 'source', 'date'.
            Defaults to None.
    
    Returns:
        DocumentResult: Processing result containing document ID, status, and metadata.
            - document_id (str): Unique identifier for the processed document
            - status (ProcessingStatus): Current processing status (PROCESSING, COMPLETED, FAILED)
            - metadata (Dict[str, Any]): Extracted and provided metadata
            - processing_time (float): Time taken to process the document in seconds
            - error_message (Optional[str]): Error details if processing failed
    
    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        UnsupportedFormatError: If the file format is not supported.
        CollectionNotFoundError: If the specified collection_id does not exist.
        ProcessingError: If document processing fails due to content or system issues.
        ValidationError: If the document fails validation checks.
    
    Example:
        >>> result = process_document(
        ...     file_path="/path/to/research_paper.pdf",
        ...     collection_id="academic_papers",
        ...     metadata={"author": "Dr. Smith", "year": 2024}
        ... )
        >>> print(f"Document ID: {result.document_id}")
        Document ID: doc_abc123
        
        >>> # Handle processing status
        >>> if result.status == ProcessingStatus.COMPLETED:
        ...     print("Document processed successfully")
        ... elif result.status == ProcessingStatus.FAILED:
        ...     print(f"Processing failed: {result.error_message}")
    
    Note:
        Large documents (>50MB) may take several minutes to process. Consider using
        the async version `process_document_async` for better performance with
        large files or batch processing.
        
        The function automatically detects file format based on file extension and
        MIME type. For best results, ensure files have correct extensions.
    
    See Also:
        process_document_async: Asynchronous version of this function
        validate_document: Pre-processing document validation
        DocumentProcessor: Low-level document processing class
    """
```

#### Class Docstrings

```python
class DocumentProcessor:
    """High-performance document processing engine for RAG systems.
    
    The DocumentProcessor handles the complete document ingestion pipeline including
    content extraction, text preprocessing, embedding generation, and vector storage.
    Supports multiple file formats and provides both synchronous and asynchronous
    processing capabilities.
    
    This class implements a modular architecture with pluggable processors for
    different file types, configurable text preprocessing pipelines, and optimized
    batch processing for large document collections.
    
    Attributes:
        supported_formats (Set[str]): File formats supported by the processor.
            Currently includes: 'pdf', 'docx', 'txt', 'md', 'html', 'rtf'.
        max_file_size (int): Maximum file size in bytes (default: 100MB).
        batch_size (int): Optimal batch size for processing multiple documents.
        embedding_model (str): Name of the embedding model used for vector generation.
        preprocessing_config (PreprocessingConfig): Configuration for text preprocessing.
    
    Example:
        >>> # Initialize processor with custom configuration
        >>> config = ProcessorConfig(
        ...     max_file_size=50 * 1024 * 1024,  # 50MB limit
        ...     batch_size=10,
        ...     embedding_model="text-embedding-ada-002"
        ... )
        >>> processor = DocumentProcessor(config)
        
        >>> # Process single document
        >>> result = processor.process_document("/path/to/document.pdf")
        >>> print(f"Status: {result.status}")
        
        >>> # Process multiple documents
        >>> documents = ["/path/to/doc1.pdf", "/path/to/doc2.docx"]
        >>> results = processor.process_batch(documents)
        >>> successful = [r for r in results if r.status == ProcessingStatus.COMPLETED]
        >>> print(f"Successfully processed: {len(successful)}/{len(results)}")
    
    Note:
        The processor maintains internal state for performance optimization including
        embedding model caching and preprocessing pipeline initialization. For
        multi-threaded usage, create separate instances or use the thread-safe
        methods marked with `_thread_safe` suffix.
        
        Memory usage scales with batch_size and document size. Monitor system
        resources when processing large document collections.
    
    Version:
        Added in version 1.0.0
        Enhanced batch processing in version 1.2.0
        Added async support in version 1.3.0
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize the document processor with configuration.
        
        Args:
            config (Optional[ProcessorConfig]): Processor configuration settings.
                If None, uses default configuration with standard settings.
                
        Raises:
            ConfigurationError: If the provided configuration is invalid.
            ModelNotFoundError: If the specified embedding model is not available.
        """
```

#### Module Docstrings

```python
"""Document Processing Module

Comprehensive document processing pipeline for the MCP Academic RAG Server.
This module provides high-level interfaces for document ingestion, content
extraction, preprocessing, and vector storage operations.

Key Components:
    - DocumentProcessor: Main processing engine
    - FileTypeDetector: Automatic file format detection  
    - ContentExtractor: Text and metadata extraction
    - PreprocessingPipeline: Text cleaning and normalization
    - EmbeddingGenerator: Vector representation generation
    - VectorStore: Document storage and retrieval

Supported Formats:
    - PDF: Including scanned documents with OCR
    - Microsoft Word: .docx, .doc formats
    - Plain Text: .txt, .md, .rst formats
    - HTML: Web pages and documentation
    - Rich Text: .rtf format

Usage Example:
    >>> from document_processing import DocumentProcessor, ProcessorConfig
    >>> 
    >>> # Initialize with custom configuration
    >>> config = ProcessorConfig(
    ...     max_file_size=100 * 1024 * 1024,  # 100MB
    ...     enable_ocr=True,
    ...     preprocessing_level="aggressive"
    ... )
    >>> processor = DocumentProcessor(config)
    >>> 
    >>> # Process document
    >>> result = processor.process_document("research_paper.pdf")
    >>> if result.success:
    ...     print(f"Document processed: {result.document_id}")

Performance Considerations:
    - Large documents (>50MB) require significant memory
    - OCR processing adds substantial processing time
    - Batch processing provides better throughput for multiple documents
    - Consider using async methods for I/O bound operations

Configuration:
    Default configuration can be overridden via environment variables:
    - PROCESSOR_MAX_FILE_SIZE: Maximum file size in bytes
    - PROCESSOR_BATCH_SIZE: Default batch processing size
    - PROCESSOR_ENABLE_OCR: Enable OCR for scanned documents
    - PROCESSOR_EMBEDDING_MODEL: Embedding model name

Dependencies:
    - PyPDF2: PDF processing
    - python-docx: Word document processing
    - pytesseract: OCR functionality (optional)
    - numpy: Numerical computations
    - transformers: Embedding model integration

Author: MCP Academic RAG Server Team
Version: 1.3.0
License: MIT
Last Updated: 2024-01-15
"""
```

### Type Annotations Standards

#### Comprehensive Type Hints

```python
from typing import (
    Dict, List, Optional, Union, Any, Callable, Iterator, 
    AsyncIterator, TypeVar, Generic, Protocol, Literal
)
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# Type aliases for clarity
DocumentID = str
CollectionID = str
EmbeddingVector = List[float]
Metadata = Dict[str, Any]

# Generic types
T = TypeVar('T')
ProcessorType = TypeVar('ProcessorType', bound='BaseProcessor')

# Protocol definitions
class Processor(Protocol):
    """Protocol for document processors."""
    
    def process(self, content: str) -> ProcessedDocument:
        """Process document content."""
        ...
    
    def supports_format(self, file_extension: str) -> bool:
        """Check if processor supports file format."""
        ...

# Dataclass with type annotations
@dataclass
class ProcessingResult:
    """Result of document processing operation."""
    
    document_id: DocumentID
    status: Literal['completed', 'failed', 'processing']
    metadata: Metadata
    processing_time: float
    error_message: Optional[str] = None
    created_at: datetime = datetime.now()

# Function with comprehensive type annotations
def query_documents(
    query: str,
    collection_id: CollectionID,
    *,
    max_results: int = 10,
    similarity_threshold: float = 0.7,
    filters: Optional[Dict[str, Union[str, int, bool]]] = None,
    include_metadata: bool = True,
    timeout: Optional[float] = None
) -> Iterator[SearchResult]:
    """Query documents with comprehensive type safety."""
```

### Error Documentation Standards

```python
class DocumentProcessingError(Exception):
    """Base exception for document processing errors.
    
    This exception is raised when document processing operations fail due to
    various reasons including file format issues, content problems, or system
    resource limitations.
    
    Attributes:
        message (str): Human-readable error message
        error_code (str): Machine-readable error code for programmatic handling
        details (Dict[str, Any]): Additional error context and debugging information
        document_path (Optional[str]): Path to the document that caused the error
        timestamp (datetime): When the error occurred
    
    Error Codes:
        - UNSUPPORTED_FORMAT: File format not supported
        - FILE_TOO_LARGE: Document exceeds size limits
        - CORRUPTED_FILE: File is corrupted or unreadable
        - EXTRACTION_FAILED: Content extraction failed
        - EMBEDDING_ERROR: Vector embedding generation failed
        - STORAGE_ERROR: Vector storage operation failed
    
    Example:
        >>> try:
        ...     result = processor.process_document("invalid.xyz")
        ... except DocumentProcessingError as e:
        ...     if e.error_code == "UNSUPPORTED_FORMAT":
        ...         print(f"Format not supported: {e.details['format']}")
        ...         print(f"Supported formats: {e.details['supported_formats']}")
        ...     else:
        ...         print(f"Processing failed: {e.message}")
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        document_path: Optional[str] = None
    ):
        """Initialize document processing error.
        
        Args:
            message: Human-readable error description
            error_code: Machine-readable error identifier
            details: Additional error context
            document_path: Path to problematic document
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.document_path = document_path
        self.timestamp = datetime.now()
```

## Code Documentation Standards

### Inline Comments

```python
def preprocess_text(text: str, config: PreprocessingConfig) -> str:
    """Preprocess text content for embedding generation."""
    
    # Remove excessive whitespace while preserving paragraph structure
    # This helps maintain semantic meaning while reducing noise
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Apply language-specific preprocessing based on detected language
    # Different languages may require different normalization strategies
    language = detect_language(text)
    if language in config.language_specific_rules:
        text = apply_language_rules(text, language, config)
    
    # Normalize Unicode characters to handle various encodings
    # This ensures consistent character representation across documents
    text = unicodedata.normalize('NFKC', text)
    
    # Apply domain-specific preprocessing for academic content
    if config.academic_mode:
        # Preserve mathematical notation and citations
        text = preserve_academic_notation(text)
        # Standardize reference formats
        text = normalize_citations(text)
    
    return text
```

### Complex Algorithm Documentation

```python
def similarity_search_with_score(
    self, 
    query_embedding: EmbeddingVector,
    k: int = 10,
    score_threshold: float = 0.0
) -> List[Tuple[Document, float]]:
    """Perform similarity search with confidence scores.
    
    Algorithm Overview:
    1. Normalize query embedding to unit vector
    2. Compute cosine similarity with all stored embeddings
    3. Apply score threshold filtering
    4. Return top-k results sorted by similarity score
    
    Mathematical Foundation:
        Cosine similarity = (A · B) / (||A|| × ||B||)
        Where A is query embedding and B is document embedding
        
        For unit vectors: similarity = A · B (dot product)
        
    Complexity:
        Time: O(n × d) where n = number of documents, d = embedding dimension
        Space: O(k) for result storage
    
    Performance Optimizations:
        - Pre-normalized embeddings stored in database
        - SIMD operations for vectorized computation
        - Early termination when score_threshold applied
        - Memory-mapped file access for large collections
    """
    
    # Normalize query embedding to unit vector for cosine similarity
    # ||v|| = sqrt(sum(v_i^2)) for i in all dimensions
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        raise ValueError("Query embedding cannot be zero vector")
    
    normalized_query = query_embedding / query_norm
    
    # Vectorized similarity computation using numpy operations
    # Takes advantage of BLAS optimizations for large matrix operations
    similarities = np.dot(self.embedding_matrix, normalized_query)
    
    # Apply threshold filtering before sorting for performance
    # Only consider documents above minimum similarity threshold
    if score_threshold > 0:
        valid_indices = np.where(similarities >= score_threshold)[0]
        filtered_similarities = similarities[valid_indices]
    else:
        valid_indices = np.arange(len(similarities))
        filtered_similarities = similarities
    
    # Get top-k results using partial sort (more efficient than full sort)
    # argpartition has O(n) average complexity vs O(n log n) for full sort
    if len(filtered_similarities) > k:
        top_k_indices = np.argpartition(filtered_similarities, -k)[-k:]
        # Sort only the top-k elements
        sorted_indices = top_k_indices[np.argsort(filtered_similarities[top_k_indices])][::-1]
    else:
        sorted_indices = np.argsort(filtered_similarities)[::-1]
    
    # Map back to original document indices
    original_indices = valid_indices[sorted_indices]
    
    # Construct result tuples with documents and scores
    results = [
        (self.documents[idx], float(similarities[idx]))
        for idx in original_indices[:k]
    ]
    
    return results
```

## Documentation Generation Standards

### Sphinx Configuration

```python
# docs/conf.py

"""Sphinx configuration for MCP Academic RAG Server documentation."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = 'MCP Academic RAG Server'
copyright = '2024, MCP Academic RAG Server Team'
author = 'MCP Academic RAG Server Team'
version = '1.3.0'
release = '1.3.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',           # Automatic documentation from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx.ext.viewcode',          # Add source code links
    'sphinx.ext.napoleon',          # Google/NumPy style docstrings
    'sphinx.ext.intersphinx',       # Link to other documentation
    'sphinx.ext.todo',              # TODO items
    'sphinx.ext.coverage',          # Documentation coverage
    'sphinx.ext.mathjax',           # Mathematical notation
    'sphinx_rtd_theme',             # Read the Docs theme
    'myst_parser',                  # Markdown support
]

# AutoDoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Napoleon configuration for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# HTML theme configuration
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-XXXXXXX-1',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'fastapi': ('https://fastapi.tiangolo.com/', None),
}
```

### API Documentation Templates

```python
# tools/generate_api_docs.py

"""API documentation generation tool."""

import inspect
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict

def generate_api_docs(module_path: str, output_path: str):
    """Generate comprehensive API documentation from source code.
    
    This tool automatically extracts docstrings, type annotations, and
    examples to create complete API documentation in multiple formats.
    
    Args:
        module_path: Path to Python module to document
        output_path: Directory to write documentation files
        
    Generated Files:
        - api_reference.md: Complete API reference
        - openapi.json: OpenAPI specification
        - examples/: Code examples directory
    """
    
    module = importlib.import_module(module_path)
    
    # Extract all public classes and functions
    api_items = []
    for name, obj in inspect.getmembers(module):
        if not name.startswith('_'):
            if inspect.isclass(obj) or inspect.isfunction(obj):
                api_items.append({
                    'name': name,
                    'type': 'class' if inspect.isclass(obj) else 'function',
                    'docstring': inspect.getdoc(obj),
                    'signature': str(inspect.signature(obj)),
                    'source_file': inspect.getfile(obj),
                    'line_number': inspect.getsourcelines(obj)[1]
                })
    
    # Generate markdown documentation
    generate_markdown_docs(api_items, output_path)
    
    # Generate OpenAPI specification
    generate_openapi_spec(api_items, output_path)
    
    # Extract and organize examples
    extract_examples(api_items, output_path)
```

## Quality Assurance

### Documentation Testing

```python
# tests/test_documentation.py

"""Documentation quality and completeness tests."""

import ast
import inspect
import pytest
from pathlib import Path
from typing import List, Set

class DocumentationChecker:
    """Automated documentation quality checker."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.required_sections = {
            'functions': ['Args', 'Returns', 'Raises', 'Example'],
            'classes': ['Attributes', 'Example'],
            'modules': ['Overview', 'Key Components', 'Usage Example']
        }
    
    def check_docstring_completeness(self) -> List[str]:
        """Check all docstrings for required sections."""
        violations = []
        
        for py_file in self.project_root.rglob("*.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            violations.extend(self._check_file_docstrings(py_file))
        
        return violations
    
    def _check_file_docstrings(self, file_path: Path) -> List[str]:
        """Check docstrings in a single Python file."""
        violations = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):  # Public functions only
                    violations.extend(
                        self._check_function_docstring(node, file_path)
                    )
            elif isinstance(node, ast.ClassDef):
                violations.extend(
                    self._check_class_docstring(node, file_path)
                )
        
        return violations

def test_all_public_functions_have_docstrings():
    """Test that all public functions have docstrings."""
    checker = DocumentationChecker(Path(__file__).parent.parent)
    violations = checker.check_docstring_completeness()
    
    if violations:
        pytest.fail(f"Documentation violations found:\n" + "\n".join(violations))

def test_docstring_format_compliance():
    """Test that docstrings follow Google style format."""
    # Implementation for format checking
    pass

def test_type_annotations_present():
    """Test that all functions have proper type annotations."""
    # Implementation for type annotation checking
    pass
```

### Continuous Documentation

```yaml
# .github/workflows/docs.yml

name: Documentation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install sphinx sphinx-rtd-theme myst-parser
    
    - name: Check documentation quality
      run: |
        python -m pytest tests/test_documentation.py -v
    
    - name: Build documentation
      run: |
        cd docs
        make html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

This comprehensive documentation system ensures:
- Consistent docstring formats across the codebase
- Automated documentation generation and validation
- Professional API documentation with examples
- Type safety through comprehensive annotations
- Quality assurance through automated testing