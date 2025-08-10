"""
OCR处理器单元测试
"""

import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import json

from processors.ocr_processor import OCRProcessor
from models.document import Document
from models.process_result import ProcessResult
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
    
    def test_structured_extraction_tables(self, ocr_processor):
        """
        测试表格结构化提取功能
        """
        # 创建包含表格的测试文档
        doc = Document("test_table.pdf")
        doc.file_path = "test_table.pdf"
        doc.file_type = "pdf"
        
        # 模拟OCR结果包含表格数据
        mock_ocr_result = {
            "status": "success",
            "text": "Table 1: Sample Data\nColumn A\tColumn B\tColumn C\nValue 1\tValue 2\tValue 3\nValue 4\tValue 5\tValue 6",
            "pages": [
                {
                    "page_num": 1,
                    "text": "Table 1: Sample Data\nColumn A\tColumn B\tColumn C\nValue 1\tValue 2\tValue 3\nValue 4\tValue 5\tValue 6"
                }
            ]
        }
        
        # 模拟结构处理器的表格识别结果
        with patch.object(ocr_processor.connector, 'process_document_ocr', return_value=mock_ocr_result):
            with patch('os.path.exists', return_value=True):
                # 处理文档
                result = ocr_processor.process_pdf_file(doc)
                
                # 验证结果包含表格数据
                assert result.is_successful()
                assert "Table 1" in result.data["text"]
                assert "Column A" in result.data["text"]
                assert "Value 1" in result.data["text"]
                
                # 验证数据结构
                assert "pages" in result.data
                assert len(result.data["pages"]) == 1
                assert result.data["pages"][0]["page_num"] == 1
    
    def test_structured_extraction_figures(self, ocr_processor):
        """
        测试图表结构化提取功能
        """
        # 创建包含图表的测试文档
        doc = Document("test_figure.pdf")
        doc.file_path = "test_figure.pdf"
        doc.file_type = "pdf"
        
        # 模拟OCR结果包含图表数据
        mock_ocr_result = {
            "status": "success",
            "text": "Figure 1: Performance Comparison Chart\nThis figure shows the performance comparison between different algorithms. The chart displays accuracy values on the y-axis and different methods on the x-axis.",
            "pages": [
                {
                    "page_num": 1,
                    "text": "Figure 1: Performance Comparison Chart\nThis figure shows the performance comparison between different algorithms."
                }
            ]
        }
        
        # 模拟结构处理器的图表识别结果
        with patch.object(ocr_processor.connector, 'process_document_ocr', return_value=mock_ocr_result):
            with patch('os.path.exists', return_value=True):
                # 处理文档
                result = ocr_processor.process_pdf_file(doc)
                
                # 验证结果包含图表数据
                assert result.is_successful()
                assert "Figure 1" in result.data["text"]
                assert "Performance Comparison" in result.data["text"]
                assert "algorithms" in result.data["text"]
                
                # 验证数据结构
                assert "pages" in result.data
                assert len(result.data["pages"]) == 1
    
    def test_multimodal_content_integration(self, ocr_processor):
        """
        测试多模态内容集成功能
        """
        # 创建包含多种元素的测试文档
        doc = Document("test_multimodal.pdf")
        doc.file_path = "test_multimodal.pdf"
        doc.file_type = "pdf"
        
        # 模拟OCR结果包含多种元素
        mock_ocr_result = {
            "status": "success",
            "text": "Research Paper Title\n\nAbstract\nThis paper presents a comprehensive study...\n\nTable 1: Experimental Results\nMethod\tAccuracy\tPrecision\nMethod A\t0.85\t0.82\nMethod B\t0.90\t0.88\n\nFigure 1: Architecture Diagram\nThe figure shows the system architecture with three main components.\n\nConclusion\nIn this work, we demonstrated...",
            "pages": [
                {
                    "page_num": 1,
                    "text": "Research Paper Title\n\nAbstract\nThis paper presents a comprehensive study..."
                },
                {
                    "page_num": 2,
                    "text": "Table 1: Experimental Results\nMethod\tAccuracy\tPrecision\nMethod A\t0.85\t0.82\nMethod B\t0.90\t0.88"
                },
                {
                    "page_num": 3,
                    "text": "Figure 1: Architecture Diagram\nThe figure shows the system architecture with three main components."
                }
            ]
        }
        
        with patch.object(ocr_processor.connector, 'process_document_ocr', return_value=mock_ocr_result):
            with patch('os.path.exists', return_value=True):
                # 处理文档
                result = ocr_processor.process_pdf_file(doc)
                
                # 验证结果包含所有类型的内容
                assert result.is_successful()
                
                # 验证文本内容
                text_content = result.data["text"]
                assert "Research Paper Title" in text_content
                assert "Abstract" in text_content
                assert "Table 1" in text_content
                assert "Figure 1" in text_content
                assert "Architecture Diagram" in text_content
                
                # 验证结构化数据
                assert "text_by_page" in result.data
                assert len(result.data["text_by_page"]) == 3
                
                # 验证各页内容
                pages = result.data["text_by_page"]
                assert "Research Paper Title" in pages[0]
                assert "Table 1" in pages[1]
                assert "Figure 1" in pages[2]
    
    def test_error_handling_structured_extraction(self, ocr_processor):
        """
        测试结构化提取的错误处理
        """
        # 创建测试文档
        doc = Document("test_error.pdf")
        doc.file_path = "test_error.pdf"
        doc.file_type = "pdf"
        
        # 模拟OCR API返回错误
        mock_error_result = {
            "status": "error",
            "message": "OCR processing failed",
            "error_code": "OCR_ERROR"
        }
        
        with patch.object(ocr_processor.connector, 'process_document_ocr', return_value=mock_error_result):
            with patch('os.path.exists', return_value=True):
                # 处理文档
                result = ocr_processor.process_pdf_file(doc)
                
                # 验证错误处理
                assert not result.is_successful()
                assert "OCR processing failed" in result.get_message()
    
    def test_data_format_validation(self, ocr_processor):
        """
        测试数据格式验证
        """
        # 创建测试文档
        doc = Document("test_format.pdf")
        doc.file_path = "test_format.pdf"
        doc.file_type = "pdf"
        
        # 模拟正常OCR结果
        mock_ocr_result = {
            "status": "success",
            "text": "Sample text with structured content.",
            "pages": [
                {"page_num": 1, "text": "Sample text with structured content."}
            ]
        }
        
        with patch.object(ocr_processor.connector, 'process_document_ocr', return_value=mock_ocr_result):
            with patch('os.path.exists', return_value=True):
                # 处理文档
                result = ocr_processor.process_pdf_file(doc)
                
                # 验证数据格式
                assert result.is_successful()
                assert isinstance(result.data, dict)
                assert "text" in result.data
                assert "pages" in result.data
                assert "text_by_page" in result.data
                
                # 验证数据类型
                assert isinstance(result.data["text"], str)
                assert isinstance(result.data["pages"], list)
                assert isinstance(result.data["text_by_page"], list)
                
                # 验证页面数据结构
                if result.data["pages"]:
                    page = result.data["pages"][0]
                    assert "page_num" in page
                    assert "text" in page
                    assert isinstance(page["page_num"], int)
                    assert isinstance(page["text"], str)
