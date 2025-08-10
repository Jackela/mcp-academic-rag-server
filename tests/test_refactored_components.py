"""
测试重构后的组件

验证重构后的核心组件是否正常工作，包括：
- 配置管理和验证
- 动态处理器加载
- 文本处理统一化
- RAG管道集成
"""

import pytest
import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.config_manager import ConfigManager
from core.config_validator import ConfigValidator, generate_default_config
from utils.text_utils import DocumentStructureExtractor
from rag.chat_session import ChatSessionManager, ChatSession
from models.document import Document


class TestConfigManagement:
    """测试配置管理功能"""
    
    def test_config_validator_creation(self):
        """测试配置验证器创建"""
        validator = ConfigValidator()
        assert validator is not None
        assert hasattr(validator, 'validate_config')
    
    def test_default_config_generation(self):
        """测试默认配置生成"""
        config = generate_default_config()
        assert isinstance(config, dict)
        assert 'storage' in config
        assert 'processors' in config
        assert 'rag_settings' in config
        assert 'llm' in config
    
    def test_processor_name_normalization(self):
        """测试处理器名称标准化"""
        validator = ConfigValidator()
        
        old_config = {
            "processors": {
                "PreProcessor": {"enabled": True},
                "OCRProcessor": {"enabled": True},
                "StructureProcessor": {"enabled": True}
            }
        }
        
        normalized = validator.normalize_processor_config(old_config)
        
        assert "pre_processor" in normalized["processors"]
        assert "ocr_processor" in normalized["processors"] 
        assert "structure_processor" in normalized["processors"]
        assert "PreProcessor" not in normalized["processors"]
    
    def test_config_validation(self):
        """测试配置验证"""
        validator = ConfigValidator()
        
        # 有效配置
        valid_config = generate_default_config()
        assert validator.validate_config(valid_config) == True
        
        # 无效配置 - 缺少必需字段
        invalid_config = {"invalid": "config"}
        assert validator.validate_config(invalid_config) == False
    
    def test_config_manager_with_temp_file(self):
        """测试配置管理器与临时文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = generate_default_config()
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config_manager = ConfigManager(temp_path)
            assert config_manager.get_value('storage.base_path') == './data'
            assert config_manager.get_value('processors.pre_processor.enabled') == True
        finally:
            os.unlink(temp_path)


class TestDocumentStructureExtractor:
    """测试文档结构提取器"""
    
    def test_extractor_creation(self):
        """测试提取器创建"""
        extractor = DocumentStructureExtractor()
        assert extractor is not None
    
    def test_structure_extraction(self):
        """测试结构提取功能"""
        sample_text = """
        Title: Test Document
        
        Abstract: This is a test document.
        
        1. Introduction
        This is the introduction section.
        
        2. Methodology
        This describes the methods used.
        """
        
        result = DocumentStructureExtractor.extract_structure(sample_text)
        
        assert isinstance(result, dict)
        assert 'title' in result
        assert 'abstract' in result
        assert 'sections' in result
        
        # 验证提取的内容
        assert 'Test Document' in result['title']
        assert 'test document' in result['abstract']
        assert len(result['sections']) >= 2


class TestChatSessionManagement:
    """测试聊天会话管理"""
    
    def test_session_manager_creation(self):
        """测试会话管理器创建"""
        manager = ChatSessionManager()
        assert manager is not None
        assert hasattr(manager, 'create_session')
        assert hasattr(manager, 'get_session')
    
    def test_session_creation(self):
        """测试会话创建"""
        manager = ChatSessionManager()
        session = manager.create_session()
        
        assert isinstance(session, ChatSession)
        assert session.session_id is not None
        assert len(session.messages) == 0
    
    def test_session_message_handling(self):
        """测试会话消息处理"""
        manager = ChatSessionManager()
        session = manager.create_session()
        
        # 添加用户消息
        user_message = session.add_message("user", "Hello, this is a test message")
        assert user_message.role == "user"
        assert user_message.content == "Hello, this is a test message"
        assert len(session.messages) == 1
        
        # 添加助手消息
        assistant_message = session.add_message("assistant", "Hello! How can I help you?")
        assert assistant_message.role == "assistant"
        assert len(session.messages) == 2
    
    @patch('rag.haystack_pipeline.RAGPipeline')
    def test_session_with_mock_rag(self, mock_rag_pipeline):
        """测试带有模拟RAG管道的会话"""
        # 设置模拟RAG管道
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {
            'answer': 'This is a mock response',
            'documents': [
                {'content': 'Mock document content', 'metadata': {'source': 'test'}}
            ]
        }
        
        manager = ChatSessionManager(rag_pipeline=mock_pipeline)
        session = manager.create_session()
        session.set_rag_pipeline(mock_pipeline)
        
        # 处理查询
        message, docs = session.process_query("What is machine learning?")
        
        assert message.role == "assistant"
        assert "mock response" in message.content.lower()
        assert len(docs) == 1


class TestDocumentProcessing:
    """测试文档处理功能"""
    
    def test_document_creation(self):
        """测试文档对象创建"""
        # 创建临时文件用于测试
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is test content")
            temp_path = f.name
        
        try:
            doc = Document(temp_path)
            assert doc.file_path == temp_path
            assert doc.file_name == os.path.basename(temp_path)
            assert doc.document_id is not None
        finally:
            os.unlink(temp_path)
    
    def test_document_content_storage(self):
        """测试文档内容存储"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            doc = Document(temp_path)
            
            # 存储处理阶段的内容
            doc.store_content("OCRProcessor", "Extracted text content")
            doc.store_content("StructureProcessor", {"title": "Test", "sections": []})
            
            # 验证内容存储
            assert doc.get_content("OCRProcessor") == "Extracted text content"
            assert isinstance(doc.get_content("StructureProcessor"), dict)
            assert doc.get_content("NonExistent") is None
        finally:
            os.unlink(temp_path)


class TestIntegration:
    """集成测试"""
    
    def test_config_processor_loading_simulation(self):
        """测试配置和处理器加载的模拟"""
        # 模拟配置
        config = {
            "processors": {
                "pre_processor": {"enabled": True, "config": {}},
                "ocr_processor": {"enabled": True, "config": {}},
                "structure_processor": {"enabled": True, "config": {}},
                "embedding_processor": {"enabled": False, "config": {}}
            }
        }
        
        # 模拟处理器加载逻辑
        enabled_processors = []
        for name, cfg in config["processors"].items():
            if cfg.get("enabled", False):
                enabled_processors.append(name)
        
        assert "pre_processor" in enabled_processors
        assert "ocr_processor" in enabled_processors
        assert "structure_processor" in enabled_processors
        assert "embedding_processor" not in enabled_processors
        assert len(enabled_processors) == 3
    
    def test_end_to_end_workflow_simulation(self):
        """测试端到端工作流程模拟"""
        # 1. 配置管理
        config = generate_default_config()
        validator = ConfigValidator()
        assert validator.validate_config(config)
        
        # 2. 文档创建
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Title: Academic Paper\n\nAbstract: This is a research paper.")
            temp_path = f.name
        
        try:
            doc = Document(temp_path)
            
            # 3. 文本结构提取
            text_content = "Title: Academic Paper\n\nAbstract: This is a research paper."
            structure = DocumentStructureExtractor.extract_structure(text_content)
            doc.store_content("StructureProcessor", structure)
            
            # 4. 会话管理
            session_manager = ChatSessionManager()
            session = session_manager.create_session()
            session.add_message("user", "What is this paper about?")
            
            # 验证整个流程
            assert doc.get_content("StructureProcessor") is not None
            assert structure['title'] is not None
            assert len(session.messages) == 1
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])