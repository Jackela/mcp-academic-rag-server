# MCP Academic RAG Server - Enhanced Developer Guide

Advanced developer guide covering the enhanced architecture, enterprise-level features, and comprehensive development practices for the MCP Academic RAG Server.

## Table of Contents

1. [Enhanced Architecture Overview](#enhanced-architecture-overview)
2. [Advanced Component Development](#advanced-component-development)
3. [Extension and Plugin System](#extension-and-plugin-system)
4. [Performance Optimization](#performance-optimization)
5. [Testing Framework](#testing-framework)
6. [Deployment and Operations](#deployment-and-operations)
7. [Contributing Guidelines](#contributing-guidelines)

## Enhanced Architecture Overview

### Enterprise-Level Architecture

The enhanced system implements a comprehensive enterprise architecture while maintaining the streamlined MCP core:

```
┌─────────────────────────────────────────────────────────┐
│                 MCP Academic RAG Server                 │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ MCP Server  │  │ FastAPI     │  │ WebSocket   │      │
│  │ (Core)      │  │ REST API    │  │ Real-time   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
├─────────────────────────────────────────────────────────┤
│                 Enhanced Core Services                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Config      │  │ Performance │  │ Workflow    │      │
│  │ Center      │  │ Monitor     │  │ Generator   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Document    │  │ Vector      │  │ LLM         │      │
│  │ Processor   │  │ Store       │  │ Connectors  │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
├─────────────────────────────────────────────────────────┤
│                  Monitoring & Observability             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Telemetry   │  │ Dashboard   │  │ Alerting    │      │
│  │ Integration │  │ System      │  │ System      │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### Enhanced Configuration System

The configuration center provides enterprise-level configuration management:

```python
from core.config_center import ConfigCenter
from core.config_runtime_validator import RuntimeConfigValidator
from core.config_version_manager import ConfigVersionManager

# Initialize configuration center with hot-reload
config_center = ConfigCenter(
    base_config_path="./config",
    environment="production",
    watch_changes=True
)

# Configuration validation
validator = RuntimeConfigValidator(ValidationLevel.ENTERPRISE)
validation_result = validator.validate_config(config_center.get_config())

if not validation_result.is_valid:
    logger.error(f"Configuration validation failed: {validation_result.errors}")

# Version management
version_manager = ConfigVersionManager(config_center.config_dir)
version_id, changes = version_manager.save_config_with_version(
    config_center.get_config(),
    description="Update LLM model configuration",
    user="admin"
)
```

### Advanced Document Processing Pipeline

```python
from core.pipeline import Pipeline
from core.processor_loader import ProcessorLoader
from models.document import Document

class EnhancedDocumentProcessor:
    """Enhanced document processor with pluggable architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor_loader = ProcessorLoader(config.get('processors', {}))
        self.pipeline = Pipeline(config)
        
        # Load and configure processors
        self.processors = self._load_processors()
    
    def _load_processors(self) -> Dict[str, Any]:
        """Load processors based on configuration."""
        processors = {}
        
        for processor_name, processor_config in self.config.get('processors', {}).items():
            if processor_config.get('enabled', False):
                processor = self.processor_loader.load_processor(
                    processor_name, 
                    processor_config
                )
                processors[processor_name] = processor
        
        return processors
    
    async def process_document(self, document: Document) -> ProcessResult:
        """Process document through enhanced pipeline."""
        try:
            # Pre-processing validation
            validation_result = await self._validate_document(document)
            if not validation_result.is_valid:
                return ProcessResult(
                    success=False,
                    message=f"Document validation failed: {validation_result.errors}"
                )
            
            # Execute processing pipeline
            result = await self.pipeline.process_document(document)
            
            # Post-processing quality checks
            quality_result = await self._check_processing_quality(document, result)
            if not quality_result.passed:
                logger.warning(f"Quality check warnings: {quality_result.warnings}")
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return ProcessResult(
                success=False,
                error=e,
                message=f"Processing failed: {str(e)}"
            )
    
    async def _validate_document(self, document: Document) -> ValidationResult:
        """Validate document before processing."""
        # Implementation for document validation
        pass
    
    async def _check_processing_quality(self, document: Document, 
                                      result: ProcessResult) -> QualityResult:
        """Check quality of processing results."""
        # Implementation for quality checks
        pass
```

## Advanced Component Development

### Custom Vector Store Implementation

```python
from document_stores.base_vector_store import BaseVectorStore
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

class CustomVectorStore(BaseVectorStore):
    """Enterprise-grade vector store with advanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dimension = config['dimension']
        self.index_type = config.get('index_type', 'faiss_ivf')
        self.distance_metric = config.get('distance_metric', 'cosine')
        
        # Initialize with configuration
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize vector store with optimizations."""
        if self.index_type == 'faiss_ivf':
            self._initialize_faiss_ivf()
        elif self.index_type == 'faiss_hnsw':
            self._initialize_faiss_hnsw()
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def _initialize_faiss_ivf(self):
        """Initialize FAISS IVF index for large-scale search."""
        import faiss
        
        nlist = min(4096, max(1, int(np.sqrt(self.config.get('max_vectors', 100000)))))
        
        if self.distance_metric == 'cosine':
            # Use inner product with normalized vectors for cosine similarity
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        # Configure search parameters
        self.index.nprobe = min(nlist, 32)  # Search 32 clusters by default
    
    async def add_vectors_batch(self, vectors: List[List[float]], 
                               metadata: List[Dict[str, Any]], 
                               batch_size: int = 1000) -> List[str]:
        """Add vectors in batches for better performance."""
        vector_ids = []
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            
            batch_ids = await self.add_vectors(batch_vectors, batch_metadata)
            vector_ids.extend(batch_ids)
            
            # Allow other coroutines to run
            await asyncio.sleep(0)
        
        return vector_ids
    
    async def search_with_filters(self, query_vector: List[float], 
                                 k: int = 10,
                                 filters: Optional[Dict[str, Any]] = None,
                                 score_threshold: float = 0.0) -> List[Tuple[str, float, Dict]]:
        """Advanced search with metadata filtering."""
        # Pre-filter if index supports it
        if filters and hasattr(self.index, 'set_selector'):
            selector = self._create_selector(filters)
            self.index.set_selector(selector)
        
        # Perform vector search
        scores, indices = await self._vector_search(query_vector, k * 2)  # Get more for filtering
        
        # Post-filter results
        filtered_results = []
        for score, idx in zip(scores, indices):
            if score < score_threshold:
                continue
                
            metadata = self._get_metadata(idx)
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            vector_id = self._get_vector_id(idx)
            filtered_results.append((vector_id, float(score), metadata))
            
            if len(filtered_results) >= k:
                break
        
        return filtered_results
    
    def _create_selector(self, filters: Dict[str, Any]):
        """Create FAISS selector for pre-filtering."""
        # Implementation for creating FAISS selectors
        pass
    
    def _matches_filters(self, metadata: Dict[str, Any], 
                        filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, condition in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(condition, dict):
                if 'gte' in condition and metadata[key] < condition['gte']:
                    return False
                if 'lte' in condition and metadata[key] > condition['lte']:
                    return False
                if 'in' in condition and metadata[key] not in condition['in']:
                    return False
            else:
                if metadata[key] != condition:
                    return False
        
        return True
```

### Advanced LLM Connector with Retry Logic

```python
from connectors.base_llm_connector import BaseLLMConnector
from typing import AsyncIterator, Dict, Any, List
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

class EnhancedOpenAIConnector(BaseLLMConnector):
    """Enhanced OpenAI connector with advanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key']
        self.model = config.get('model', 'gpt-4')
        self.max_retries = config.get('max_retries', 3)
        self.timeout = config.get('timeout', 30)
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(config.get('max_concurrent', 10))
        
        # Session management
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response_with_retry(self, prompt: str, 
                                         context: List[str], 
                                         **kwargs) -> str:
        """Generate response with automatic retry logic."""
        async with self.rate_limiter:
            return await self._generate_response_internal(prompt, context, **kwargs)
    
    async def _generate_response_internal(self, prompt: str, 
                                        context: List[str], 
                                        **kwargs) -> str:
        """Internal response generation with error handling."""
        messages = self._build_messages(prompt, context)
        
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': kwargs.get('max_tokens', 1000),
            'temperature': kwargs.get('temperature', 0.7),
            'stream': False
        }
        
        async with self.session.post(
            'https://api.openai.com/v1/chat/completions',
            json=payload
        ) as response:
            if response.status == 429:
                # Rate limited - wait and retry
                await asyncio.sleep(2 ** self.current_retry)
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=429,
                    message="Rate limited"
                )
            
            response.raise_for_status()
            data = await response.json()
            
            return data['choices'][0]['message']['content']
    
    async def stream_response_with_backpressure(self, prompt: str, 
                                              context: List[str],
                                              **kwargs) -> AsyncIterator[str]:
        """Stream response with backpressure handling."""
        messages = self._build_messages(prompt, context)
        
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': kwargs.get('max_tokens', 1000),
            'temperature': kwargs.get('temperature', 0.7),
            'stream': True
        }
        
        async with self.rate_limiter:
            async with self.session.post(
                'https://api.openai.com/v1/chat/completions',
                json=payload
            ) as response:
                response.raise_for_status()
                
                buffer = ""
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        buffer += content
                                        
                                        # Yield complete words/sentences
                                        if content.endswith((' ', '.', '!', '?', '\n')):
                                            yield buffer
                                            buffer = ""
                            except json.JSONDecodeError:
                                continue
                
                # Yield remaining buffer
                if buffer:
                    yield buffer
```

## Extension and Plugin System

### Plugin Architecture

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import importlib
import inspect

class BasePlugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.enabled = config.get('enabled', True)
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize plugin. Return True if successful."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown plugin gracefully. Return True if successful."""
        pass
    
    def get_hooks(self) -> List[str]:
        """Return list of hooks this plugin implements."""
        hooks = []
        for method_name in dir(self):
            if method_name.startswith('on_') and callable(getattr(self, method_name)):
                hooks.append(method_name)
        return hooks

class PluginManager:
    """Manage plugin loading, initialization, and execution."""
    
    def __init__(self, plugin_config: Dict[str, Any]):
        self.plugin_config = plugin_config
        self.plugins: Dict[str, BasePlugin] = {}
        self.hooks: Dict[str, List[BasePlugin]] = {}
    
    async def load_plugins(self):
        """Load and initialize all configured plugins."""
        for plugin_name, config in self.plugin_config.items():
            if not config.get('enabled', False):
                continue
            
            try:
                plugin = await self._load_plugin(plugin_name, config)
                if plugin and await plugin.initialize():
                    self.plugins[plugin_name] = plugin
                    self._register_hooks(plugin)
                    logger.info(f"Plugin loaded successfully: {plugin_name}")
                else:
                    logger.error(f"Failed to initialize plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_name}: {e}")
    
    async def _load_plugin(self, plugin_name: str, config: Dict[str, Any]) -> Optional[BasePlugin]:
        """Load a single plugin."""
        module_path = config.get('module', f'plugins.{plugin_name}')
        class_name = config.get('class', f'{plugin_name.title()}Plugin')
        
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            
            if not issubclass(plugin_class, BasePlugin):
                raise ValueError(f"Plugin class {class_name} must inherit from BasePlugin")
            
            return plugin_class(config)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return None
    
    def _register_hooks(self, plugin: BasePlugin):
        """Register plugin hooks."""
        for hook_name in plugin.get_hooks():
            if hook_name not in self.hooks:
                self.hooks[hook_name] = []
            self.hooks[hook_name].append(plugin)
    
    async def execute_hook(self, hook_name: str, *args, **kwargs):
        """Execute all plugins for a specific hook."""
        if hook_name in self.hooks:
            for plugin in self.hooks[hook_name]:
                if plugin.enabled:
                    try:
                        hook_method = getattr(plugin, hook_name)
                        if inspect.iscoroutinefunction(hook_method):
                            await hook_method(*args, **kwargs)
                        else:
                            hook_method(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error executing hook {hook_name} on plugin {plugin.name}: {e}")

# Example plugin implementation
class MetricsPlugin(BasePlugin):
    """Plugin for collecting custom metrics."""
    
    async def initialize(self) -> bool:
        """Initialize metrics collection."""
        self.metrics_endpoint = self.config.get('endpoint', 'http://localhost:9090/metrics')
        self.collection_interval = self.config.get('interval', 60)
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown metrics collection."""
        return True
    
    async def on_document_processed(self, document: Document, result: ProcessResult):
        """Hook: Called when document processing completes."""
        await self._record_metric(
            'documents_processed_total',
            1,
            tags={'status': 'success' if result.success else 'error'}
        )
    
    async def on_query_executed(self, query: str, results: List[Any], duration: float):
        """Hook: Called when query is executed."""
        await self._record_metric(
            'query_duration_seconds',
            duration,
            tags={'result_count': str(len(results))}
        )
    
    async def _record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record metric to external system."""
        # Implementation for metric recording
        pass
```

### Custom Processor Development

```python
from processors.base_processor import IProcessor
from models.document import Document
from models.process_result import ProcessResult
from typing import Dict, Any

class CustomOCRProcessor(IProcessor):
    """Advanced OCR processor with multiple engine support."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine = config.get('engine', 'tesseract')
        self.languages = config.get('languages', ['eng'])
        self.confidence_threshold = config.get('confidence_threshold', 60)
        
        # Initialize OCR engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize OCR engine based on configuration."""
        if self.engine == 'tesseract':
            self._initialize_tesseract()
        elif self.engine == 'easyocr':
            self._initialize_easyocr()
        elif self.engine == 'cloud_vision':
            self._initialize_cloud_vision()
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")
    
    def _initialize_tesseract(self):
        """Initialize Tesseract OCR."""
        try:
            import pytesseract
            from PIL import Image
            
            self.tesseract = pytesseract
            self.Image = Image
            
            # Configure Tesseract
            config = '--oem 3 --psm 6'  # LSTM OCR engine, uniform text block
            if self.confidence_threshold:
                config += f' -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
            
            self.tesseract_config = config
            
        except ImportError:
            raise ImportError("pytesseract not installed. Install with: pip install pytesseract")
    
    async def process(self, document: Document) -> ProcessResult:
        """Process document with OCR."""
        try:
            # Check if document needs OCR
            if not self._needs_ocr(document):
                return ProcessResult(
                    success=True,
                    message="Document does not require OCR processing"
                )
            
            # Extract images from document
            images = await self._extract_images(document)
            if not images:
                return ProcessResult(
                    success=True,
                    message="No images found for OCR processing"
                )
            
            # Process images with OCR
            ocr_results = []
            for i, image in enumerate(images):
                try:
                    text = await self._perform_ocr(image)
                    if text.strip():
                        ocr_results.append({
                            'page': i + 1,
                            'text': text,
                            'confidence': self._calculate_confidence(text)
                        })
                except Exception as e:
                    logger.warning(f"OCR failed for image {i}: {e}")
            
            # Store OCR results in document
            document.content['ocr_results'] = ocr_results
            document.metadata['ocr_engine'] = self.engine
            document.metadata['ocr_languages'] = self.languages
            
            total_text = '\n'.join([result['text'] for result in ocr_results])
            
            return ProcessResult(
                success=True,
                message=f"OCR processing completed. Extracted {len(total_text)} characters from {len(ocr_results)} pages",
                data={
                    'extracted_text': total_text,
                    'pages_processed': len(ocr_results),
                    'engine': self.engine
                }
            )
            
        except Exception as e:
            return ProcessResult(
                success=False,
                message=f"OCR processing failed: {str(e)}",
                error=e
            )
    
    def _needs_ocr(self, document: Document) -> bool:
        """Determine if document needs OCR processing."""
        # Check if document is image-based (PDF with images, image files)
        file_extension = Path(document.file_path).suffix.lower()
        
        # Image files always need OCR
        if file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return True
        
        # For PDFs, check if they contain text or are image-based
        if file_extension == '.pdf':
            return self._is_image_pdf(document.file_path)
        
        return False
    
    async def _extract_images(self, document: Document) -> List[Any]:
        """Extract images from document for OCR processing."""
        images = []
        file_extension = Path(document.file_path).suffix.lower()
        
        if file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            # Single image file
            image = self.Image.open(document.file_path)
            images.append(image)
        
        elif file_extension == '.pdf':
            # Extract images from PDF
            images = await self._extract_pdf_images(document.file_path)
        
        return images
    
    async def _perform_ocr(self, image) -> str:
        """Perform OCR on a single image."""
        if self.engine == 'tesseract':
            return self.tesseract.image_to_string(
                image,
                lang='+'.join(self.languages),
                config=self.tesseract_config
            )
        elif self.engine == 'easyocr':
            # EasyOCR implementation
            result = self.easyocr_reader.readtext(np.array(image))
            return ' '.join([text[1] for text in result])
        elif self.engine == 'cloud_vision':
            # Google Cloud Vision implementation
            return await self._cloud_vision_ocr(image)
        
        return ""
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted text."""
        # Simple heuristic based on text characteristics
        if not text.strip():
            return 0.0
        
        # Count alphanumeric characters vs total
        alphanumeric_count = sum(c.isalnum() for c in text)
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return 0.0
        
        confidence = (alphanumeric_count / total_chars) * 100
        return min(confidence, 100.0)
```

This enhanced developer guide provides comprehensive coverage of the advanced architecture, enterprise features, and development patterns for the MCP Academic RAG Server. The documentation focuses on practical implementation examples and best practices for extending the system.