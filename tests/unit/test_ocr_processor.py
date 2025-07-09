"""
OCR处理器单元测试
"""

import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import json

from processors.ocr_processor import OCRProcessor
from models.document import Document
from core.process_result import ProcessResult
from connectors.api_connector import MistralAPIConnector


class TestOCRProcessor:
    """OCR处理器单元测试类"""
    
    @pytest.fixture
    def mock_ocr_connector(self):
        """模拟OCR API连接器"""
        connector = MagicMock(spec=MistralAPIConnector)
        connector.process_document_ocr.return_value = {
            "status": "success",
            "text": "OCR extracted text",
            "pages": [
                {"page_num": 1, "text": "Page 1 text"},
                {"page_num": 2, "text": "Page 2 text"}
            ]
        }
        connector.process_image_ocr.return_value = {
            "status": "success",
            "text": "OCR extracted text from image",
            "pages": [{"page_num": 1, "text": "Image text"}]
        }
        return connector
    
    @pytest.fixture
    def ocr_processor(self, mock_ocr_connector):
        """创建OCR处理器实例"""
        with patch('processors.ocr_processor.OCRAPIFactory') as mock_factory:
            mock_factory.create_connector.return_value = mock_ocr_connector
            processor = OCRProcessor()
            processor.api_type = "mistral"
            processor.api_config = {
                "api_key": "test-key",
                "api_url": "https://api.test.com"
            }
            processor.initialize()
            return processor
    
    def test_initialization(self, mock_ocr_connector):
        """测试初始化过程"""
        with patch('processors.ocr_processor.OCRAPIFactory') as mock_factory:
            mock_factory.create_connector.return_value = mock_ocr_connector
            
            processor = OCRProcessor()
            processor.api_type = "mistral"
            processor.api_config = {
                "api_key": "test-key",
                "api_url": "https://api.test.com"
            }
            processor.initialize()
            
            # 验证API工厂被正确调用
            mock_factory.create_connector.assert_called_once_with(
                "mistral", 
                processor.api_config
            )
            
            # 验证连接器设置正确
            assert processor.connector == mock_ocr_connector
    
    def test_process_pdf_file(self, ocr_processor, mock_ocr_connector):
        """测试处理PDF文件"""
        # 创建测试文档
        doc = Document("test.pdf")
        doc.file_path = "test.pdf"
        doc.file_type = "pdf"
        
        # 模拟文件存在检查
        with patch('os.path.exists', return_value=True):
            # 处理文档
            result = ocr_processor.process_pdf_file(doc)
            
            # 验证API调用
            mock_ocr_connector.process_document_ocr.assert_called_once()
            
            # 验证结果
            assert isinstance(result, ProcessResult)
            assert result.is_successful()
            assert result.data["text"] == "OCR extracted text"
            assert len(result.data["pages"]) == 2
    
    def test_process_image_file(self, ocr_processor, mock_ocr_connector):
        """测试处理图像文件"""
        # 创建测试文档
        doc = Document("test.jpg")
        doc.file_path = "test.jpg"
        doc.file_type = "image"
        
        # 模拟文件存在检查
        with patch('os.path.exists', return_value=True):
            # 处理文档
            result = ocr_processor.process_image_file(doc)
            
            # 验证API调用
            mock_ocr_connector.process_image_ocr.assert_called_once_with("test.jpg")
            
            # 验证结果
            assert isinstance(result, ProcessResult)
            assert result.is_successful()
            assert result.data["text"] == "OCR extracted text from image"
    
    def test_process_document(self, ocr_processor):
        """测试处理文档（主方法）"""
        # 创建测试文档
        doc = Document("test.pdf")
        doc.file_path = "test.pdf"
        doc.file_type = "pdf"
        
        # 模拟方法调用
        with patch.object(ocr_processor, 'process_pdf_file') as mock_process_pdf:
            mock_process_pdf.return_value = ProcessResult(
                success=True,
                message="OCR处理成功",
                data={"text": "PDF OCR 文本"}
            )
            
            # 处理文档
            result = ocr_processor.process(doc)
            
            # 验证方法调用
            mock_process_pdf.assert_called_once_with(doc)
            
            # 验证结果
            assert result.is_successful()
            assert result.get_message() == "OCR处理成功"
            assert "text" in result.data
    
    def test_process_unsupported_file(self, ocr_processor):
        """测试处理不支持的文件类型"""
        # 创建不支持的文件类型的文档
        doc = Document("test.doc")
        doc.file_path = "test.doc"
        doc.file_type = "doc"  # 不支持的类型
        
        # 处理文档
        result = ocr_processor.process(doc)
        
        # 验证结果
        assert not result.is_successful()
        assert "不支持的文件类型" in result.get_message()
    
    def test_combine_page_results(self, ocr_processor):
        """测试合并页面结果"""
        # 创建测试页面结果
        page_results = [
            {"page_num": 1, "text": "Page 1 text"},
            {"page_num": 2, "text": "Page 2 text"},
            {"page_num": 3, "text": "Page 3 text"}
        ]
        
        # 合并结果
        combined_text = ocr_processor._combine_page_results(page_results)
        
        # 验证结果
        assert "Page 1 text" in combined_text
        assert "Page 2 text" in combined_text
        assert "Page 3 text" in combined_text
        
        # 验证页面分隔符
        for i in range(1, 3):  # 检查页面1和2之后是否有分隔符
            assert f"Page {i} text\n\n------ 第 {i} 页结束 ------\n\n" in combined_text
