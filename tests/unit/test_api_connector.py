"""
API连接器单元测试
"""

import pytest
from unittest.mock import patch, MagicMock
import json
import requests
from connectors.api_connector import APIConnector, MistralAPIConnector, OpenAIAPIConnector, APIFactory


class TestAPIConnector:
    """API连接器单元测试类"""
    
    @pytest.fixture
    def mock_response(self):
        """模拟API响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {'content-type': 'application/json'}
        mock_resp.json.return_value = {"status": "success", "result": "test_result"}
        return mock_resp
    
    @patch('requests.request')
    def test_make_request_success(self, mock_request, mock_response):
        """测试成功的API请求"""
        # 设置模拟响应
        mock_request.return_value = mock_response
        
        # 创建测试连接器
        class TestConnector(APIConnector):
            def _build_headers(self):
                return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        connector = TestConnector("https://api.test.com", "test-api-key")
        
        # 发送请求
        result = connector.make_request("GET", "test-endpoint", params={"param": "value"})
        
        # 验证结果
        assert result == {"status": "success", "result": "test_result"}
        
        # 验证请求参数
        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.test.com/test-endpoint",
            headers={"Authorization": "Bearer test-api-key", "Content-Type": "application/json"},
            params={"param": "value"},
            data=None,
            files=None,
            json=None,
            timeout=60
        )
    
    @patch('requests.request')
    def test_make_request_retry(self, mock_request):
        """测试请求重试机制"""
        # 设置前两次请求失败，第三次成功
        mock_error_response = MagicMock()
        mock_error_response.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
        mock_error_response.text = "Error response"
        
        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.headers = {'content-type': 'application/json'}
        mock_success_response.json.return_value = {"status": "success", "result": "retry_success"}
        
        mock_request.side_effect = [
            mock_error_response,
            mock_error_response,
            mock_success_response
        ]
        
        # 创建测试连接器
        class TestConnector(APIConnector):
            def _build_headers(self):
                return {"Authorization": f"Bearer {self.api_key}"}
        
        connector = TestConnector("https://api.test.com", "test-api-key")
        
        # 减少重试延迟以加快测试
        result = connector.make_request("GET", "test-endpoint", retry_delay=0.01)
        
        # 验证结果
        assert result == {"status": "success", "result": "retry_success"}
        
        # 验证请求被调用了三次
        assert mock_request.call_count == 3


class TestMistralAPIConnector:
    """Mistral API连接器单元测试类"""
    
    @patch.object(MistralAPIConnector, 'make_request')
    def test_process_document_ocr(self, mock_make_request):
        """测试文档OCR处理"""
        # 设置模拟响应
        mock_make_request.return_value = {
            "status": "success",
            "text": "OCR extracted text",
            "pages": [{"page_num": 1, "text": "Page 1 text"}]
        }
        
        # 创建连接器
        connector = MistralAPIConnector("https://api.mistral.ai", "test-api-key")
        
        # 调用OCR处理
        result = connector.process_document_ocr("https://example.com/document.pdf")
        
        # 验证结果
        assert result["status"] == "success"
        assert "text" in result
        
        # 验证请求参数
        mock_make_request.assert_called_once_with(
            "POST", 
            "ocr/process", 
            json_data={
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": "https://example.com/document.pdf"
                },
                "include_image_base64": False
            }
        )
    
    @patch.object(MistralAPIConnector, 'process_document_ocr')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('base64.b64encode')
    def test_process_image_ocr(self, mock_b64encode, mock_open, mock_process_document_ocr):
        """测试图像OCR处理"""
        # 设置模拟响应
        mock_process_document_ocr.return_value = {
            "status": "success",
            "text": "OCR extracted text from image"
        }
        
        # 模拟文件读取和base64编码
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.read.return_value = b"image_binary_data"
        mock_b64encode.return_value = b"encoded_image_data"
        
        # 创建连接器
        connector = MistralAPIConnector("https://api.mistral.ai", "test-api-key")
        
        # 调用图像OCR处理
        result = connector.process_image_ocr("test_image.jpg")
        
        # 验证结果
        assert result["status"] == "success"
        assert result["text"] == "OCR extracted text from image"
        
        # 验证文件操作
        mock_open.assert_called_once_with("test_image.jpg", "rb")
        mock_file.read.assert_called_once()
        mock_b64encode.assert_called_once_with(b"image_binary_data")
        
        # 验证处理调用
        mock_process_document_ocr.assert_called_once_with(
            "data:image/jpeg;base64,encoded_image_data",
            "mistral-ocr-latest",
            include_image=False
        )


class TestOpenAIAPIConnector:
    """OpenAI API连接器单元测试类"""
    
    @patch.object(OpenAIAPIConnector, 'make_request')
    def test_classify_document(self, mock_make_request):
        """测试文档分类"""
        # 设置模拟响应
        mock_make_request.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"category": "science", "confidence": 0.95})
                    }
                }
            ]
        }
        
        # 创建连接器
        connector = OpenAIAPIConnector("https://api.openai.com/v1", "test-api-key")
        
        # 调用文档分类
        result = connector.classify_document(
            "This is a scientific paper about quantum physics.",
            categories=["science", "art", "history"]
        )
        
        # 验证结果
        assert result["category"] == "science"
        assert result["confidence"] == 0.95
        
        # 验证请求参数
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        assert args[0] == "POST"
        assert args[1] == "chat/completions"
        assert "model" in kwargs["json_data"]
        assert kwargs["json_data"]["model"] == "gpt-4"
        assert "messages" in kwargs["json_data"]


class TestAPIFactory:
    """API工厂类单元测试"""
    
    def test_create_connector_mistral(self):
        """测试创建Mistral连接器"""
        config = {
            "api_url": "https://api.mistral.ai",
            "api_key": "test-mistral-key",
            "timeout": 30
        }
        
        connector = APIFactory.create_connector("mistral", config)
        
        assert isinstance(connector, MistralAPIConnector)
        assert connector.api_url == "https://api.mistral.ai"
        assert connector.api_key == "test-mistral-key"
        assert connector.timeout == 30
    
    def test_create_connector_openai(self):
        """测试创建OpenAI连接器"""
        config = {
            "api_url": "https://api.openai.com/v1",
            "api_key": "test-openai-key",
            "timeout": 45
        }
        
        connector = APIFactory.create_connector("openai", config)
        
        assert isinstance(connector, OpenAIAPIConnector)
        assert connector.api_url == "https://api.openai.com/v1"
        assert connector.api_key == "test-openai-key"
        assert connector.timeout == 45
    
    def test_create_connector_unsupported(self):
        """测试创建不支持的连接器类型"""
        config = {
            "api_url": "https://api.test.com",
            "api_key": "test-key"
        }
        
        with pytest.raises(ValueError, match="不支持的API类型"):
            APIFactory.create_connector("unsupported", config)
