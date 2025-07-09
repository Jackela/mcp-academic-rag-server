"""
API连接器模块 - 提供统一的外部API接口
"""

import os
import time
import json
import requests
import logging
import base64
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, BinaryIO

# 配置日志
logger = logging.getLogger(__name__)

class APIConnector(ABC):
    """API连接器基类"""
    
    def __init__(self, api_url: str, api_key: str, timeout: int = 60):
        """
        初始化API连接器
        
        Args:
            api_url (str): API端点URL
            api_key (str): API密钥
            timeout (int): 请求超时时间(秒)
        """
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.headers = self._build_headers()
        
    @abstractmethod
    def _build_headers(self) -> Dict[str, str]:
        """构建API请求头"""
        pass
    
    def make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                    data: Optional[Dict] = None, files: Optional[Dict] = None, 
                    json_data: Optional[Dict] = None, max_retries: int = 3, 
                    retry_delay: int = 2) -> Dict[str, Any]:
        """
        发送API请求并处理响应
        
        Args:
            method (str): 请求方法 (GET, POST, PUT, DELETE)
            endpoint (str): API端点路径
            params (dict, optional): URL参数
            data (dict, optional): 表单数据
            files (dict, optional): 文件数据
            json_data (dict, optional): JSON数据
            max_retries (int): 最大重试次数
            retry_delay (int): 重试延迟(秒)
            
        Returns:
            dict: API响应数据
            
        Raises:
            Exception: 请求失败时抛出
        """
        url = f"{self.api_url.rstrip('/')}/{endpoint.lstrip('/')}"
        current_try = 0
        
        while current_try < max_retries:
            current_try += 1
            try:
                logger.debug(f"发送{method}请求到: {url}")
                
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    params=params,
                    data=data,
                    files=files,
                    json=json_data,
                    timeout=self.timeout
                )
                
                # 检查HTTP状态码
                response.raise_for_status()
                
                if response.headers.get('content-type') == 'application/json':
                    return response.json()
                else:
                    return {"status": "success", "text": response.text}
                
            except requests.RequestException as e:
                logger.error(f"API请求失败({current_try}/{max_retries}): {e}")
                
                # 如果达到最大重试次数，则抛出异常
                if current_try >= max_retries:
                    logger.error(f"API请求失败，已达到最大重试次数: {url}")
                    raise Exception(f"API请求失败: {str(e)}")
                
                # 获取响应内容（如果有）
                response_text = None
                response_json = None
                
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        response_text = e.response.text
                        try:
                            response_json = e.response.json()
                        except:
                            pass
                except:
                    pass
                
                # 记录错误响应
                if response_json:
                    logger.error(f"API错误响应: {json.dumps(response_json, ensure_ascii=False)}")
                elif response_text:
                    logger.error(f"API错误响应: {response_text}")
                
                # 等待后重试
                time.sleep(retry_delay)
                
class MistralAPIConnector(APIConnector):
    """Mistral AI API连接器"""
    
    def _build_headers(self) -> Dict[str, str]:
        """构建Mistral API请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    def process_document_ocr(self, document_url: str, model: str = "mistral-ocr-latest", 
                           include_image: bool = False) -> Dict[str, Any]:
        """
        使用Mistral OCR API处理文档
        
        Args:
            document_url (str): 文档URL或Base64编码
            model (str): OCR模型名称
            include_image (bool): 是否包含图像数据
            
        Returns:
            dict: OCR处理结果
        """
        endpoint = "ocr/process"
        payload = {
            "model": model,
            "document": {
                "type": "document_url" if not document_url.startswith("data:") else "image_url",
                "document_url" if not document_url.startswith("data:") else "image_url": document_url
            },
            "include_image_base64": include_image
        }
        
        logger.info(f"开始OCR处理文档: {model}")
        return self.make_request("POST", endpoint, json_data=payload)
    
    def process_image_ocr(self, image_path: str, model: str = "mistral-ocr-latest") -> Dict[str, Any]:
        """
        使用Mistral OCR API处理图像
        
        Args:
            image_path (str): 图像文件路径
            model (str): OCR模型名称
            
        Returns:
            dict: OCR处理结果
        """
        # 将图像编码为Base64
        try:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            # 获取图像MIME类型
            image_ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.tiff': 'image/tiff',
                '.bmp': 'image/bmp'
            }.get(image_ext, 'image/jpeg')
            
            # 创建data URI
            image_url = f"data:{mime_type};base64,{encoded_image}"
            
            # 调用OCR处理
            return self.process_document_ocr(image_url, model, include_image=False)
            
        except Exception as e:
            logger.error(f"处理图像OCR失败: {image_path}, 错误: {str(e)}")
            raise
    
    def upload_file(self, file_path: str, purpose: str = "ocr") -> Dict[str, Any]:
        """
        上传文件到Mistral AI
        
        Args:
            file_path (str): 文件路径
            purpose (str): 文件用途
            
        Returns:
            dict: 上传响应
        """
        endpoint = "files/upload"
        
        try:
            file_name = os.path.basename(file_path)
            files = {
                "file": (file_name, open(file_path, "rb"), "application/octet-stream")
            }
            
            data = {
                "purpose": purpose
            }
            
            logger.info(f"开始上传文件: {file_name}")
            response = self.make_request("POST", endpoint, data=data, files=files)
            logger.info(f"文件上传成功: {file_name}")
            return response
            
        except Exception as e:
            logger.error(f"文件上传失败: {file_path}, 错误: {str(e)}")
            raise
    
    def get_signed_url(self, file_id: str) -> Dict[str, Any]:
        """
        获取文件的签名URL
        
        Args:
            file_id (str): 文件ID
            
        Returns:
            dict: 包含签名URL的响应
        """
        endpoint = f"files/get_signed_url/{file_id}"
        
        logger.info(f"获取文件签名URL: {file_id}")
        return self.make_request("GET", endpoint)


class OpenAIAPIConnector(APIConnector):
    """OpenAI API连接器"""
    
    def _build_headers(self) -> Dict[str, str]:
        """构建OpenAI API请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def classify_document(self, content: str, categories: List[str] = None, 
                         model: str = "gpt-4") -> Dict[str, Any]:
        """
        使用OpenAI API进行文档分类
        
        Args:
            content (str): 文档内容
            categories (List[str], optional): 分类类别列表
            model (str): 使用的模型名称
            
        Returns:
            dict: 分类结果
        """
        endpoint = "chat/completions"
        
        # 构建系统提示
        system_prompt = "你是一个专业的学术文献分类助手，负责对文档进行分类、提取关键词和主题。"
        
        # 构建用户提示
        user_prompt = "请分析以下学术文献内容，并提供以下信息：\n"
        user_prompt += "1. 文档类别：" 
        
        if categories and len(categories) > 0:
            user_prompt += f"请从[{', '.join(categories)}]中选择最合适的类别。\n"
        else:
            user_prompt += "请提供你认为最准确的类别。\n"
            
        user_prompt += "2. 关键词：提取5-10个最能代表文档内容的关键词或术语。\n"
        user_prompt += "3. 主题：确定1-3个文档的主要主题或研究领域。\n"
        user_prompt += "4. 摘要：用100-200字概括文档的主要内容和贡献。\n\n"
        user_prompt += "请以JSON格式返回结果，包含以下字段：categories(数组)、keywords(数组)、topics(数组)、summary(字符串)。\n\n"
        user_prompt += f"文档内容：\n{content}"
        
        # 构建请求
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        logger.info(f"开始分类文档: {model}")
        result = self.make_request("POST", endpoint, json_data=payload)
        
        # 提取分类结果
        try:
            response_content = result["choices"][0]["message"]["content"]
            return json.loads(response_content)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"解析分类结果失败: {str(e)}")
            return {}
    
    def extract_topics(self, content: str, model: str = "gpt-4") -> Dict[str, Any]:
        """
        使用OpenAI API提取文档主题和关键信息
        
        Args:
            content (str): 文档内容
            model (str): 使用的模型名称
            
        Returns:
            dict: 主题提取结果
        """
        endpoint = "chat/completions"
        
        # 构建系统提示
        system_prompt = "你是一个专业的学术内容分析专家，负责从文档中提取主题、关键信息和知识见解。"
        
        # 构建用户提示
        user_prompt = "请从以下学术文献内容中提取以下信息：\n"
        user_prompt += "1. 主要主题：识别3-5个核心主题或概念\n"
        user_prompt += "2. 关键发现：提取文档中的主要研究发现或结论\n"
        user_prompt += "3. 方法论：识别使用的研究方法或技术\n"
        user_prompt += "4. 关键引用：识别重要的引用和参考文献\n"
        user_prompt += "5. 潜在应用：分析研究的可能应用或影响\n\n"
        user_prompt += "请以JSON格式返回结果，包含以下字段：topics(数组)、findings(数组)、methodologies(数组)、key_citations(数组)、applications(数组)。\n\n"
        user_prompt += f"文档内容：\n{content}"
        
        # 构建请求
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        logger.info(f"开始提取主题: {model}")
        result = self.make_request("POST", endpoint, json_data=payload)
        
        # 提取结果
        try:
            response_content = result["choices"][0]["message"]["content"]
            return json.loads(response_content)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"解析主题提取结果失败: {str(e)}")
            return {}
    
    def enhance_document_structure(self, content: str, model: str = "gpt-4") -> Dict[str, Any]:
        """
        使用OpenAI API增强文档结构识别
        
        Args:
            content (str): 文档内容
            model (str): 使用的模型名称
            
        Returns:
            dict: 结构识别结果
        """
        endpoint = "chat/completions"
        
        # 构建系统提示
        system_prompt = "你是一个专业的文档结构分析专家，负责识别和提取文档的结构元素。"
        
        # 构建用户提示
        user_prompt = "请分析以下学术文献内容，提取文档的结构元素：\n"
        user_prompt += "1. 标题：提取文档的主标题\n"
        user_prompt += "2. 摘要：提取文档的摘要部分\n"
        user_prompt += "3. 章节：识别文档的所有章节标题及其层级关系\n"
        user_prompt += "4. 引言：提取引言或介绍部分\n"
        user_prompt += "5. 结论：提取结论或总结部分\n"
        user_prompt += "6. 参考文献：识别参考文献部分\n\n"
        user_prompt += "请以JSON格式返回结果，包含以下字段：title(字符串)、abstract(字符串)、introduction(字符串)、conclusion(字符串)、sections(数组，每个元素包含title和level)、references(数组)。\n\n"
        user_prompt += f"文档内容：\n{content}"
        
        # 构建请求
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        logger.info(f"开始增强文档结构识别: {model}")
        result = self.make_request("POST", endpoint, json_data=payload)
        
        # 提取结果
        try:
            response_content = result["choices"][0]["message"]["content"]
            return json.loads(response_content)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"解析结构识别结果失败: {str(e)}")
            return {}


class APIFactory:
    """API工厂类，创建不同服务的API连接器"""
    
    @staticmethod
    def create_connector(api_type: str, config: Dict[str, Any]) -> APIConnector:
        """
        创建API连接器
        
        Args:
            api_type (str): API类型，如 'openai', 'mistral', 'azure'
            config (dict): API配置
            
        Returns:
            APIConnector: API连接器实例
            
        Raises:
            ValueError: 不支持的API类型
        """
        if api_type.lower() == 'mistral':
            return MistralAPIConnector(
                api_url=config.get('api_url', 'https://api.mistral.ai/v1'),
                api_key=config.get('api_key', '')
            )
        elif api_type.lower() == 'openai':
            return OpenAIAPIConnector(
                api_url=config.get('api_url', 'https://api.openai.com/v1'),
                api_key=config.get('api_key', '')
            )
        else:
            raise ValueError(f"不支持的API类型: {api_type}")


# 保留原有的OCRAPIFactory类，为了向后兼容
class OCRAPIFactory:
    """OCR API工厂类，创建不同OCR服务的连接器"""
    
    @staticmethod
    def create_connector(api_type: str, config: Dict[str, Any]) -> APIConnector:
        """
        创建OCR API连接器
        
        Args:
            api_type (str): API类型，如 'mistral', 'azure', 'google'
            config (dict): API配置
            
        Returns:
            APIConnector: API连接器实例
            
        Raises:
            ValueError: 不支持的API类型
        """
        return APIFactory.create_connector(api_type, config)
