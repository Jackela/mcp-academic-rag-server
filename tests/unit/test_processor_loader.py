"""
Tests for core.processor_loader module

Comprehensive test suite covering ProcessorLoader class functionality including:
- Dynamic processor loading from configuration
- External configuration file handling
- Processor mapping resolution
- Error handling for missing modules/classes
- Validation and performance monitoring
"""

import pytest
import json
import tempfile
import importlib
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from typing import Dict, Any

from core.processor_loader import ProcessorLoader
from core.config_manager import ConfigManager
from processors.base_processor import IProcessor


class MockConfigManager:
    """Mock config manager for testing"""
    
    def __init__(self, config_data: Dict[str, Any] = None):
        self.config_data = config_data or {}
    
    def get_value(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        current = self.config_data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current


class MockProcessor(IProcessor):
    """Mock processor implementing IProcessor interface"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "MockProcessor"
    
    def get_name(self) -> str:
        return self.name
    
    def get_stage(self) -> str:
        return "test"
    
    def supports_file_type(self, file_type: str) -> bool:
        return True
    
    def process(self, document) -> Any:
        return Mock()


class InvalidProcessor:
    """Invalid processor that doesn't implement IProcessor interface"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}


@pytest.fixture
def sample_config():
    """Create sample configuration for testing"""
    return {
        "processors": {
            "pre_processor": {
                "enabled": True,
                "config": {"batch_size": 10}
            },
            "ocr_processor": {
                "enabled": True,
                "config": {"engine": "tesseract"}
            },
            "disabled_processor": {
                "enabled": False,
                "config": {}
            }
        },
        "processor_mappings": {
            "custom_processor": {
                "module": "custom.module",
                "class": "CustomProcessor",
                "description": "Custom test processor"
            }
        }
    }


@pytest.fixture
def mock_config_manager(sample_config):
    """Create mock config manager with sample configuration"""
    return MockConfigManager(sample_config)


class TestProcessorLoaderInitialization:
    """Test ProcessorLoader initialization"""
    
    def test_initialization_with_config_manager(self, mock_config_manager):
        """Test initialization with config manager"""
        loader = ProcessorLoader(mock_config_manager)
        
        assert loader.config_manager == mock_config_manager
        assert hasattr(loader, 'logger')
        assert hasattr(loader, '_processor_mappings')
    
    def test_initialization_loads_mappings(self, mock_config_manager):
        """Test that initialization loads processor mappings"""
        loader = ProcessorLoader(mock_config_manager)
        
        # Should have both default and custom mappings
        mappings = loader._processor_mappings
        assert "custom_processor" in mappings
        assert mappings["custom_processor"]["module"] == "custom.module"


class TestProcessorMappingLoading:
    """Test processor mapping loading strategies"""
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_from_external_file(self, mock_file, mock_exists, mock_config_manager):
        """Test loading processor mappings from external file"""
        # Setup external file mock
        mock_exists.return_value = True
        external_mappings = {
            "processor_mappings": {
                "external_processor": {
                    "module": "external.module",
                    "class": "ExternalProcessor",
                    "description": "External processor"
                }
            }
        }
        mock_file.return_value.read.return_value = json.dumps(external_mappings)
        
        loader = ProcessorLoader(mock_config_manager)
        
        assert "external_processor" in loader._processor_mappings
        assert loader._processor_mappings["external_processor"]["module"] == "external.module"
    
    @patch('pathlib.Path.exists')
    def test_fallback_to_config_manager(self, mock_exists, mock_config_manager):
        """Test fallback to config manager when external file doesn't exist"""
        mock_exists.return_value = False
        
        loader = ProcessorLoader(mock_config_manager)
        
        # Should use mappings from config manager
        assert "custom_processor" in loader._processor_mappings
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', side_effect=IOError("File read error"))
    def test_fallback_on_file_error(self, mock_file, mock_exists, mock_config_manager):
        """Test fallback when external file read fails"""
        mock_exists.return_value = True
        
        loader = ProcessorLoader(mock_config_manager)
        
        # Should fallback to config manager mappings
        assert "custom_processor" in loader._processor_mappings
    
    def test_default_mappings_fallback(self):
        """Test fallback to default mappings when no external sources available"""
        empty_config_manager = MockConfigManager({})
        
        with patch('pathlib.Path.exists', return_value=False):
            loader = ProcessorLoader(empty_config_manager)
        
        # Should have default mappings
        mappings = loader._processor_mappings
        assert "pre_processor" in mappings
        assert "ocr_processor" in mappings
        assert "embedding_processor" in mappings
        
        # Check default mapping structure
        assert mappings["pre_processor"]["module"] == "processors.pre_processor"
        assert mappings["pre_processor"]["class"] == "PreProcessor"


class TestProcessorLoading:
    """Test actual processor loading functionality"""
    
    def test_load_processors_with_enabled_processors(self, mock_config_manager):
        """Test loading enabled processors"""
        with patch.object(ProcessorLoader, '_load_single_processor') as mock_load:
            mock_processor = MockProcessor()
            mock_load.return_value = mock_processor
            
            loader = ProcessorLoader(mock_config_manager)
            processors = loader.load_processors()
            
            # Should load 2 enabled processors (pre_processor and ocr_processor)
            assert len(processors) == 2
            assert mock_load.call_count == 2
    
    def test_load_processors_skips_disabled(self, mock_config_manager):
        """Test that disabled processors are skipped"""
        with patch.object(ProcessorLoader, '_load_single_processor') as mock_load:
            mock_processor = MockProcessor()
            mock_load.return_value = mock_processor
            
            loader = ProcessorLoader(mock_config_manager)
            processors = loader.load_processors()
            
            # Should not load disabled_processor
            loaded_names = [call[0][0] for call in mock_load.call_args_list]
            assert "disabled_processor" not in loaded_names
    
    def test_load_processors_continues_on_error(self, mock_config_manager):
        """Test that loading continues when individual processor fails"""
        def side_effect(name, config):
            if name == "ocr_processor":
                raise ImportError("Failed to load ocr_processor")
            return MockProcessor()
        
        with patch.object(ProcessorLoader, '_load_single_processor', side_effect=side_effect):
            loader = ProcessorLoader(mock_config_manager)
            processors = loader.load_processors()
            
            # Should load only the successful processor
            assert len(processors) == 1
    
    def test_load_processors_performance_logging(self, mock_config_manager):
        """Test that performance metrics are logged"""
        with patch.object(ProcessorLoader, '_load_single_processor') as mock_load:
            mock_processor = MockProcessor()
            mock_load.return_value = mock_processor
            
            loader = ProcessorLoader(mock_config_manager)
            
            with patch.object(loader.logger, 'info') as mock_log:
                processors = loader.load_processors()
                
                # Should log performance information
                assert any("load_duration" in str(call) for call in mock_log.call_args_list)


class TestSingleProcessorLoading:
    """Test loading individual processors"""
    
    def test_load_single_processor_from_mapping(self, mock_config_manager):
        """Test loading processor using explicit mapping"""
        config = {"config": {"test": "value"}}
        
        # Mock the processor class
        mock_processor_class = Mock(return_value=MockProcessor())
        mock_module = Mock()
        mock_module.MockProcessor = mock_processor_class
        
        with patch('importlib.import_module', return_value=mock_module):
            loader = ProcessorLoader(mock_config_manager)
            # Add a mapping for test
            loader._processor_mappings["test_processor"] = {
                "module": "test.module",
                "class": "MockProcessor",
                "description": "Test processor"
            }
            
            processor = loader._load_single_processor("test_processor", config)
            
            assert processor is not None
            mock_processor_class.assert_called_once_with(config={"test": "value"})
    
    def test_load_single_processor_convention_based(self, mock_config_manager):
        """Test loading processor using convention-based naming"""
        config = {
            "module": "custom.module", 
            "class": "CustomClass",
            "config": {"setting": "value"}
        }
        
        # Mock the processor class
        mock_processor_class = Mock(return_value=MockProcessor())
        mock_module = Mock()
        mock_module.CustomClass = mock_processor_class
        
        with patch('importlib.import_module', return_value=mock_module):
            loader = ProcessorLoader(mock_config_manager)
            
            processor = loader._load_single_processor("unknown_processor", config)
            
            assert processor is not None
            mock_processor_class.assert_called_once_with(config={"setting": "value"})
    
    def test_load_single_processor_import_error(self, mock_config_manager):
        """Test handling of module import errors"""
        config = {"config": {}}
        
        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            loader = ProcessorLoader(mock_config_manager)
            
            with pytest.raises(ImportError, match="Failed to import processor module"):
                loader._load_single_processor("test_processor", config)
    
    def test_load_single_processor_class_not_found(self, mock_config_manager):
        """Test handling of missing processor class"""
        config = {"config": {}}
        
        mock_module = Mock()
        # Remove the expected class attribute
        del mock_module.TestProcessor
        
        with patch('importlib.import_module', return_value=mock_module):
            with patch('hasattr', return_value=False):
                loader = ProcessorLoader(mock_config_manager)
                loader._processor_mappings["test_processor"] = {
                    "module": "test.module",
                    "class": "TestProcessor"
                }
                
                with pytest.raises(AttributeError, match="Class TestProcessor not found"):
                    loader._load_single_processor("test_processor", config)
    
    def test_load_single_processor_instantiation_error(self, mock_config_manager):
        """Test handling of processor instantiation errors"""
        config = {"config": {}}
        
        # Mock processor class that raises exception during instantiation
        mock_processor_class = Mock(side_effect=TypeError("Instantiation failed"))
        mock_module = Mock()
        mock_module.TestProcessor = mock_processor_class
        
        with patch('importlib.import_module', return_value=mock_module):
            loader = ProcessorLoader(mock_config_manager)
            loader._processor_mappings["test_processor"] = {
                "module": "test.module",
                "class": "TestProcessor"
            }
            
            with pytest.raises(TypeError, match="Failed to instantiate processor"):
                loader._load_single_processor("test_processor", config)
    
    def test_load_single_processor_interface_validation(self, mock_config_manager):
        """Test that processor interface compliance is validated"""
        config = {"config": {}}
        
        # Create processor that doesn't implement IProcessor
        invalid_processor = InvalidProcessor()
        mock_processor_class = Mock(return_value=invalid_processor)
        mock_module = Mock()
        mock_module.InvalidProcessor = mock_processor_class
        
        with patch('importlib.import_module', return_value=mock_module):
            loader = ProcessorLoader(mock_config_manager)
            loader._processor_mappings["invalid_processor"] = {
                "module": "test.module",
                "class": "InvalidProcessor"
            }
            
            with patch.object(loader.logger, 'warning') as mock_warning:
                processor = loader._load_single_processor("invalid_processor", config)
                
                # Should return processor but log warning
                assert processor == invalid_processor
                mock_warning.assert_called_once()
                assert "does not implement IProcessor interface" in str(mock_warning.call_args)


class TestProcessorValidation:
    """Test processor configuration validation"""
    
    def test_validate_processor_config_valid(self, mock_config_manager):
        """Test validation of valid processor configuration"""
        with patch.object(ProcessorLoader, '_load_single_processor') as mock_load:
            mock_load.return_value = MockProcessor()
            
            loader = ProcessorLoader(mock_config_manager)
            result = loader.validate_processor_config("pre_processor")
            
            assert result is True
    
    def test_validate_processor_config_disabled(self, mock_config_manager):
        """Test validation of disabled processor"""
        loader = ProcessorLoader(mock_config_manager)
        result = loader.validate_processor_config("disabled_processor")
        
        # Disabled processors should be considered valid
        assert result is True
    
    def test_validate_processor_config_invalid(self, mock_config_manager):
        """Test validation of invalid processor configuration"""
        with patch.object(ProcessorLoader, '_load_single_processor', side_effect=ImportError("Failed")):
            loader = ProcessorLoader(mock_config_manager)
            result = loader.validate_processor_config("pre_processor")
            
            assert result is False
    
    def test_validate_processor_config_missing(self, mock_config_manager):
        """Test validation of missing processor configuration"""
        loader = ProcessorLoader(mock_config_manager)
        result = loader.validate_processor_config("nonexistent_processor")
        
        assert result is True  # Missing config with enabled=True default


class TestUtilityMethods:
    """Test utility methods"""
    
    def test_get_available_processors(self, mock_config_manager):
        """Test getting available processor information"""
        loader = ProcessorLoader(mock_config_manager)
        available = loader.get_available_processors()
        
        # Should return copy of mappings
        assert isinstance(available, dict)
        assert "custom_processor" in available
        assert available["custom_processor"]["description"] == "Custom test processor"
        
        # Verify it's a copy, not the original
        available["new_processor"] = {"test": "value"}
        assert "new_processor" not in loader._processor_mappings


class TestErrorScenarios:
    """Test various error scenarios"""
    
    def test_malformed_external_config_file(self, mock_config_manager):
        """Test handling of malformed external configuration file"""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="invalid json")):
                # Should fallback to config manager without raising exception
                loader = ProcessorLoader(mock_config_manager)
                assert "custom_processor" in loader._processor_mappings
    
    def test_empty_processor_config(self):
        """Test handling of empty processor configuration"""
        empty_config_manager = MockConfigManager({})
        
        with patch('pathlib.Path.exists', return_value=False):
            loader = ProcessorLoader(empty_config_manager)
            processors = loader.load_processors()
            
            # Should return empty list when no processors configured
            assert len(processors) == 0
    
    def test_missing_config_section(self):
        """Test handling of missing processor configuration section"""
        config_without_processors = MockConfigManager({"other_config": {}})
        
        with patch('pathlib.Path.exists', return_value=False):
            loader = ProcessorLoader(config_without_processors)
            processors = loader.load_processors()
            
            assert len(processors) == 0


class TestIntegration:
    """Integration tests"""
    
    def test_full_loading_workflow(self, mock_config_manager):
        """Test complete processor loading workflow"""
        # Mock successful processor loading
        mock_processor1 = MockProcessor()
        mock_processor1.name = "PreProcessor"
        mock_processor2 = MockProcessor()
        mock_processor2.name = "OCRProcessor"
        
        def mock_load_side_effect(name, config):
            if name == "pre_processor":
                return mock_processor1
            elif name == "ocr_processor":
                return mock_processor2
            return None
        
        with patch.object(ProcessorLoader, '_load_single_processor', side_effect=mock_load_side_effect):
            loader = ProcessorLoader(mock_config_manager)
            processors = loader.load_processors()
            
            assert len(processors) == 2
            assert processors[0].name == "PreProcessor"
            assert processors[1].name == "OCRProcessor"
    
    def test_mixed_success_failure_loading(self, mock_config_manager):
        """Test loading with mixed success and failure scenarios"""
        mock_processor = MockProcessor()
        
        def mock_load_side_effect(name, config):
            if name == "pre_processor":
                return mock_processor
            elif name == "ocr_processor":
                raise ImportError("OCR module not available")
            return None
        
        with patch.object(ProcessorLoader, '_load_single_processor', side_effect=mock_load_side_effect):
            loader = ProcessorLoader(mock_config_manager)
            processors = loader.load_processors()
            
            # Should load only successful processor
            assert len(processors) == 1
            assert processors[0] == mock_processor
    
    def test_performance_monitoring_integration(self, mock_config_manager):
        """Test that performance monitoring works end-to-end"""
        import time
        
        def slow_processor_load(name, config):
            time.sleep(0.01)  # Simulate slow loading
            return MockProcessor()
        
        with patch.object(ProcessorLoader, '_load_single_processor', side_effect=slow_processor_load):
            loader = ProcessorLoader(mock_config_manager)
            
            with patch.object(loader.logger, 'info') as mock_log:
                processors = loader.load_processors()
                
                # Check that performance information was logged
                log_calls = [str(call) for call in mock_log.call_args_list]
                
                # Should log loading completion with metrics
                completion_logs = [log for log in log_calls if "Processor loading completed" in log]
                assert len(completion_logs) > 0
                
                # Should log individual processor loading times
                duration_logs = [log for log in log_calls if "load_duration" in log]
                assert len(duration_logs) > 0