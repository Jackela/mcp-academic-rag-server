"""
Processor Loader - Dynamic processor loading with external configuration
Separates processor mapping from core server logic
"""

import importlib
import logging
from typing import List, Dict, Any
from pathlib import Path

from core.config_manager import ConfigManager
from processors.base_processor import IProcessor


class ProcessorLoader:
    """
    Dynamically loads processors based on configuration.
    
    This class handles the loading of document processors, allowing for
    flexible configuration and easy extension of processing capabilities.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the processor loader.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger("mcp-academic-rag-server.processor-loader")
        
        # Load processor mappings from configuration or use defaults
        self._processor_mappings = self._load_processor_mappings()
    
    def _load_processor_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Load processor mappings from configuration with fallback to defaults.
        
        Returns:
            Dictionary mapping processor names to their module and class information
        """
        # Try to load from external file first
        try:
            import json
            from pathlib import Path
            
            config_path = Path("config/processor_mappings.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    external_mappings = config_data.get("processor_mappings", {})
                    
                if external_mappings:
                    self.logger.info(
                        "Loaded processor mappings from external file",
                        extra={'config_file': str(config_path), 'mappings_count': len(external_mappings)}
                    )
                    return external_mappings
        except Exception as e:
            self.logger.warning(f"Could not load external processor mappings: {str(e)}")
        
        # Try to load from main configuration
        external_mappings = self.config_manager.get_value("processor_mappings", {})
        
        if external_mappings:
            self.logger.info("Loaded processor mappings from main configuration")
            return external_mappings
        
        # Fallback to default mappings
        default_mappings = {
            'pre_processor': {
                'module': 'processors.pre_processor',
                'class': 'PreProcessor',
                'description': 'Initial document preprocessing'
            },
            'ocr_processor': {
                'module': 'processors.ocr_processor', 
                'class': 'OCRProcessor',
                'description': 'Optical Character Recognition processing'
            },
            'structure_processor': {
                'module': 'processors.structure_processor',
                'class': 'StructureProcessor',
                'description': 'Document structure analysis'
            },
            'classification_processor': {
                'module': 'processors.classification_processor',
                'class': 'ClassificationProcessor',
                'description': 'Document classification and categorization'
            },
            'format_converter': {
                'module': 'processors.format_converter',
                'class': 'FormatConverterProcessor',
                'description': 'Document format conversion'
            },
            'embedding_processor': {
                'module': 'processors.haystack_embedding_processor',
                'class': 'HaystackEmbeddingProcessor',
                'description': 'Vector embedding generation'
            }
        }
        
        self.logger.info("Using default processor mappings")
        return default_mappings
    
    def load_processors(self) -> List[IProcessor]:
        """
        Load all enabled processors based on configuration with performance monitoring.
        
        Returns:
            List of loaded processor instances
            
        Raises:
            ImportError: If a required processor module cannot be imported
            AttributeError: If a processor class cannot be found
        """
        import time
        
        processors = []
        processor_configs = self.config_manager.get_value("processors", {})
        
        start_time = time.time()
        self.logger.info(f"Loading processors from {len(processor_configs)} configurations")
        
        # Track loading metrics
        success_count = 0
        error_count = 0
        
        for processor_name, processor_config in processor_configs.items():
            if not processor_config.get("enabled", True):
                self.logger.info(f"Skipping disabled processor: {processor_name}")
                continue
            
            processor_start = time.time()
            try:
                processor = self._load_single_processor(processor_name, processor_config)
                if processor:
                    processors.append(processor)
                    processor_duration = time.time() - processor_start
                    success_count += 1
                    
                    self.logger.info(
                        f"Successfully loaded processor: {processor_name}",
                        extra={
                            'processor_name': processor_name,
                            'processor_type': type(processor).__name__,
                            'load_duration': f"{processor_duration:.3f}s"
                        }
                    )
                
            except Exception as e:
                error_count += 1
                processor_duration = time.time() - processor_start
                
                self.logger.error(
                    f"Failed to load processor {processor_name}: {str(e)}",
                    extra={
                        'processor_name': processor_name, 
                        'error': str(e),
                        'load_duration': f"{processor_duration:.3f}s"
                    },
                    exc_info=True
                )
                # Continue loading other processors instead of failing completely
                continue
        
        total_duration = time.time() - start_time
        self.logger.info(
            f"Processor loading completed: {success_count} processors loaded, {error_count} failed",
            extra={
                'total_processors': len(processors),
                'success_count': success_count,
                'error_count': error_count,
                'total_duration': f"{total_duration:.3f}s",
                'avg_duration_per_processor': f"{total_duration/max(len(processor_configs), 1):.3f}s"
            }
        )
        
        return processors
    
    def _load_single_processor(self, processor_name: str, processor_config: Dict[str, Any]) -> IProcessor:
        """
        Load a single processor instance with comprehensive error handling and validation.
        
        This method implements a two-phase loading strategy:
        1. Configuration resolution: Determine module path and class name from mappings or conventions
        2. Dynamic loading: Import module, instantiate class, and validate interface compliance
        
        Args:
            processor_name: Name of the processor to load (e.g., 'ocr_processor')
            processor_config: Configuration dictionary containing processor settings
            
        Returns:
            Loaded processor instance implementing IProcessor interface
            
        Raises:
            ImportError: If the processor module cannot be imported
            AttributeError: If the processor class cannot be found in the module
            TypeError: If the processor cannot be instantiated or doesn't implement IProcessor
        """
        # Phase 1: Configuration resolution with fallback strategy
        # First, try to resolve from explicit processor mappings (preferred)
        if processor_name in self._processor_mappings:
            mapping = self._processor_mappings[processor_name]
            module_path = mapping['module']  # e.g., 'processors.ocr_processor'
            class_name = mapping['class']    # e.g., 'OCRProcessor'
            description = mapping.get('description', 'No description available')
        else:
            # Fallback to convention-based naming for dynamic processors
            # This allows loading of processors not defined in the mappings
            module_path = processor_config.get("module", f"processors.{processor_name}")
            class_name = processor_config.get("class", f"{processor_name.title()}Processor")
            description = "Dynamically loaded processor"
        
        self.logger.debug(
            f"Loading processor: {processor_name}",
            extra={
                'processor_name': processor_name,
                'module_path': module_path,
                'class_name': class_name,
                'description': description
            }
        )
        
        # Phase 2: Dynamic loading with comprehensive error handling
        
        # Step 1: Import the module using Python's importlib system
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            # Module import failed - could be missing dependencies or wrong path
            self.logger.error(f"Cannot import module {module_path}: {str(e)}")
            raise ImportError(f"Failed to import processor module {module_path}: {str(e)}")
        
        # Step 2: Retrieve the processor class from the imported module
        try:
            processor_class = getattr(module, class_name)
        except AttributeError as e:
            # Class not found in module - could be wrong class name or module structure
            self.logger.error(f"Cannot find class {class_name} in module {module_path}")
            raise AttributeError(f"Class {class_name} not found in module {module_path}: {str(e)}")
        
        # Step 3: Instantiate the processor with configuration
        try:
            # Extract processor-specific configuration, defaulting to empty dict
            processor_init_config = processor_config.get("config", {})
            
            # Create the processor instance with its configuration
            processor = processor_class(config=processor_init_config)
            
            # Step 4: Validate interface compliance
            # Ensure the loaded processor implements the IProcessor interface
            # This is critical for maintaining contract compliance across the system
            if not isinstance(processor, IProcessor):
                self.logger.warning(
                    f"Processor {processor_name} does not implement IProcessor interface. "
                    f"This may cause runtime errors during pipeline execution."
                )
            
            return processor
            
        except Exception as e:
            # Instantiation failed - could be configuration issues, missing dependencies,
            # or constructor errors in the processor class
            self.logger.error(f"Cannot instantiate processor {class_name}: {str(e)}")
            raise TypeError(f"Failed to instantiate processor {class_name}: {str(e)}")
    
    def get_available_processors(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about all available processors.
        
        Returns:
            Dictionary with processor information including descriptions
        """
        return self._processor_mappings.copy()
    
    def validate_processor_config(self, processor_name: str) -> bool:
        """
        Validate if a processor configuration is valid.
        
        Args:
            processor_name: Name of the processor to validate
            
        Returns:
            True if the processor can be loaded, False otherwise
        """
        try:
            processor_config = self.config_manager.get_value("processors", {}).get(processor_name, {})
            if not processor_config.get("enabled", True):
                return True  # Disabled processors are valid
            
            self._load_single_processor(processor_name, processor_config)
            return True
            
        except Exception as e:
            self.logger.debug(f"Processor {processor_name} validation failed: {str(e)}")
            return False