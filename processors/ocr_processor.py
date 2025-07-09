
"""
OCR处理器模块 - 实现文档OCR文本识别功能

该模块提供了OCR处理器，用于识别扫描文档中的文本内容，
支持通过Mistral AI等外部OCR API进行文本识别。
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Union

from processors.base_processor import BaseProcessor
from models.document import Document
from models.process_result import ProcessResult
from connectors.api_connector import OCRAPIFactory

# 配置日志
logger = logging.getLogger(__name__)

class OCRProcessor(BaseProcessor):
    """OCR处理器，通过外部API进行文本识别"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化OCR处理器
        
        Args:
            config (dict, optional): 处理器配置
        """
        name = "OCR处理器"
        description = "通过OCR API识别文档文本内容"
        super().__init__(name, description, config)
        
        # 默认配置
        self.default_config = {
            'api_type': 'mistral',         # OCR API类型，如mistral, azure, google
            'model': 'mistral-ocr-latest', # OCR模型
            'batch_size': 10,              # 批处理大小
            'max_retries': 3,              # 最大重试次数
            'retry_delay': 2,              # 重试延迟(秒)
            'api_config': {                # API特定配置
                'api_url': 'https://api.mistral.ai/v1',
                'api_key': ''              # API密钥应从环境变量或安全存储获取
            }
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        
        # 创建API连接器
        try:
            self.api_connector = OCRAPIFactory.create_connector(
                self.config['api_type'],
                self.config['api_config']
            )
            logger.info(f"已创建OCR API连接器: {self.config['api_type']}")
        except Exception as e:
            logger.error(f"创建OCR API连接器失败: {str(e)}")
            self.api_connector = None
    
    def get_stage(self) -> str:
        """
        获取处理阶段名称
        
        Returns:
            str: 处理阶段名称
        """
        return "ocr"
    
    def supports_file_type(self, file_type: str) -> bool:
        """
        检查是否支持此文件类型
        
        Args:
            file_type (str): 文件扩展名
            
        Returns:
            bool: 是否支持
        """
        supported_types = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.pdf']
        return file_type.lower() in supported_types
    
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档OCR
        
        Args:
            document (Document): 要处理的文档对象
            
        Returns:
            ProcessResult: 处理结果
        """
        if not self.api_connector:
            return ProcessResult.error_result("OCR API连接器未初始化")
        
        try:
            logger.info(f"开始OCR处理文档: {document.file_name}")
            
            # 更新文档状态
            document.update_status("ocr_processing")
            
            # 获取预处理结果
            pre_result = document.get_content("pre_processing")
            
            # 如果有预处理结果，使用预处理后的文件
            if pre_result and "processed_files" in pre_result and pre_result["processed_files"]:
                input_files = pre_result["processed_files"]
                logger.debug(f"使用预处理后的文件进行OCR: {len(input_files)}个文件")
            else:
                # 否则使用原始文件
                input_files = [document.file_path]
                logger.debug(f"使用原始文件进行OCR: {document.file_path}")
            
            # 进行OCR处理
            ocr_results = self._process_ocr(input_files)
            
            if not ocr_results:
                return ProcessResult.error_result("OCR处理失败，未获取到结果")
            
            # 合并多页OCR结果
            combined_text, combined_text_by_page = self._combine_ocr_results(ocr_results)
            
            # 记录处理结果
            result_data = {
                "text": combined_text,
                "text_by_page": combined_text_by_page,
                "ocr_api": self.config['api_type'],
                "ocr_model": self.config['model'],
                "raw_results": ocr_results  # 可选：保存原始OCR结果
            }
            
            # 将处理结果保存到文档对象
            document.store_content(self.get_stage(), result_data)
            document.add_metadata("has_ocr", True)
            document.add_metadata("text_length", len(combined_text))
            document.add_metadata("page_count", len(combined_text_by_page))
            
            logger.info(f"文档OCR处理完成: {document.file_name}, 文本长度: {len(combined_text)}字符")
            return ProcessResult.success_result("OCR文本识别完成", result_data)
            
        except Exception as e:
            logger.error(f"文档OCR处理失败: {document.file_name}, 错误: {str(e)}", exc_info=True)
            return ProcessResult.error_result(f"OCR处理时发生错误: {str(e)}", e)
    
    def _process_ocr(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        处理一组文件的OCR
        
        Args:
            file_paths (List[str]): 文件路径列表
            
        Returns:
            List[Dict[str, Any]]: OCR结果列表
        """
        ocr_results = []
        
        for i, file_path in enumerate(file_paths):
            logger.debug(f"处理第{i+1}/{len(file_paths)}个文件: {file_path}")
            
            try:
                # 根据文件类型确定OCR处理方法
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.pdf':
                    # PDF处理方式
                    result = self._process_pdf_ocr(file_path)
                else:
                    # 图像处理方式
                    result = self._process_image_ocr(file_path)
                
                if result:
                    ocr_results.append(result)
                    logger.debug(f"文件OCR成功: {file_path}")
                else:
                    logger.warning(f"文件OCR失败: {file_path}")
                
                # 避免API速率限制
                if i < len(file_paths) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"文件OCR处理异常: {file_path}, 错误: {str(e)}")
                # 继续处理下一个文件
        
        return ocr_results
    
    def _process_pdf_ocr(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        处理PDF文件OCR
        
        Args:
            pdf_path (str): PDF文件路径
            
        Returns:
            Optional[Dict[str, Any]]: OCR结果
        """
        # 根据OCR API类型选择处理方法
        if self.config['api_type'] == 'mistral':
            try:
                # 1. 上传文件
                upload_response = self.api_connector.upload_file(pdf_path, purpose="ocr")
                file_id = upload_response.get('id')
                
                if not file_id:
                    logger.error(f"PDF上传失败: {pdf_path}, 响应: {upload_response}")
                    return None
                
                # 2. 获取签名URL
                signed_url_response = self.api_connector.get_signed_url(file_id)
                signed_url = signed_url_response.get('url')
                
                if not signed_url:
                    logger.error(f"获取签名URL失败: {file_id}, 响应: {signed_url_response}")
                    return None
                
                # 3. 进行OCR处理
                ocr_response = self.api_connector.process_document_ocr(
                    signed_url, 
                    model=self.config['model']
                )
                
                return ocr_response
                
            except Exception as e:
                logger.error(f"Mistral PDF OCR处理失败: {pdf_path}, 错误: {str(e)}")
                return None
        else:
            logger.error(f"不支持的OCR API类型: {self.config['api_type']}")
            return None
    
    def _process_image_ocr(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        处理图像文件OCR
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            Optional[Dict[str, Any]]: OCR结果
        """
        # 根据OCR API类型选择处理方法
        if self.config['api_type'] == 'mistral':
            try:
                # 调用Mistral图像OCR API
                ocr_response = self.api_connector.process_image_ocr(
                    image_path,
                    model=self.config['model']
                )
                
                return ocr_response
                
            except Exception as e:
                logger.error(f"Mistral图像OCR处理失败: {image_path}, 错误: {str(e)}")
                return None
        else:
            logger.error(f"不支持的OCR API类型: {self.config['api_type']}")
            return None
    
    def _combine_ocr_results(self, results: List[Dict[str, Any]]) -> tuple:
        """
        合并多个OCR结果
        
        Args:
            results (List[Dict[str, Any]]): OCR结果列表
            
        Returns:
            tuple: (合并的文本, 按页面分组的文本列表)
        """
        combined_text = ""
        text_by_page = []
        
        for result in results:
            if not result:
                continue
                
            # 提取文本内容
            if self.config['api_type'] == 'mistral':
                # Mistral API返回格式处理
                if 'content' in result:
                    page_text = result['content'].get('text', '')
                    if page_text:
                        combined_text += page_text + "\n\n"
                        text_by_page.append(page_text)
                        
                # 多页文档处理
                if 'pages' in result:
                    for page in result['pages']:
                        page_text = page.get('text', '')
                        if page_text:
                            combined_text += page_text + "\n\n"
                            text_by_page.append(page_text)
            else:
                # 其他API返回格式处理
                if 'text' in result:
                    page_text = result['text']
                    combined_text += page_text + "\n\n"
                    text_by_page.append(page_text)
        
        return combined_text.strip(), text_by_page
