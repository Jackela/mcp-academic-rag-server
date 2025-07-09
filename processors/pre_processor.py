
"""
图像预处理器模块 - 实现文档图像预处理功能

该模块提供了文档预处理器，用于优化OCR前的图像质量，
包括图像增强、倾斜校正、噪点去除等功能。
"""

import os
import logging
from typing import Dict, Any, List, Tuple

from processors.base_processor import BaseProcessor
from models.document import Document
from models.process_result import ProcessResult
from utils.image_utils import ImageUtils

# 配置日志
logger = logging.getLogger(__name__)

class PreProcessor(BaseProcessor):
    """
    图像预处理器，优化OCR前的图像质量
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化预处理器
        
        Args:
            config (dict, optional): 处理器配置
        """
        name = "预处理器"
        description = "优化文档图像质量，提高OCR识别准确率"
        super().__init__(name, description, config)
        
        # 默认配置
        self.default_config = {
            'brightness': 1.0,         # 亮度调整因子
            'contrast': 1.2,           # 对比度调整因子
            'sharpness': 1.2,          # 锐度调整因子
            'deskew_enabled': True,    # 是否启用倾斜校正
            'noise_removal': 'median', # 去噪方法: median, gaussian, none
            'output_format': 'PNG',    # 输出图像格式
            'binarize': 'adaptive',    # 二值化方法: adaptive, otsu, none
            'output_directory': './processed_images/' # 处理后图像保存目录
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        
        # 创建输出目录
        os.makedirs(self.config['output_directory'], exist_ok=True)
        
        # 支持的文件类型
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.pdf']
    
    def get_stage(self) -> str:
        """
        获取处理阶段名称
        
        Returns:
            str: 处理阶段名称
        """
        return "pre_processing"
    
    def supports_file_type(self, file_type: str) -> bool:
        """
        检查是否支持此文件类型
        
        Args:
            file_type (str): 文件扩展名
            
        Returns:
            bool: 是否支持
        """
        return file_type.lower() in self.supported_extensions
    
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档图像
        
        Args:
            document (Document): 要处理的文档对象
            
        Returns:
            ProcessResult: 处理结果
        """
        try:
            logger.info(f"开始预处理文档: {document.file_name}")
            
            # 检查文件类型是否支持
            if not self.supports_file_type(document.file_type):
                return ProcessResult.error_result(
                    f"不支持的文件类型: {document.file_type}，仅支持: {', '.join(self.supported_extensions)}"
                )
            
            # 更新文档状态
            document.update_status("preprocessing")
            
            # 获取图像处理实例
            image_utils = ImageUtils()
            
            # 处理单个文件或PDF多页
            processed_files = []
            
            if document.file_type.lower() == '.pdf':
                # 处理PDF文件
                processed_files = self._process_pdf(document, image_utils)
            else:
                # 处理单个图像文件
                processed_file = self._process_image(document.file_path, image_utils)
                if processed_file:
                    processed_files.append(processed_file)
            
            if not processed_files:
                return ProcessResult.error_result("预处理失败，未生成处理后的文件")
            
            # 记录处理结果
            result_data = {
                "processed_files": processed_files,
                "processing_parameters": self.config
            }
            
            # 将处理结果保存到文档对象
            document.store_content(self.get_stage(), result_data)
            document.add_metadata("has_preprocessing", True)
            
            logger.info(f"文档预处理完成: {document.file_name}, 生成{len(processed_files)}个处理后的文件")
            return ProcessResult.success_result("图像预处理完成", result_data)
            
        except Exception as e:
            logger.error(f"文档预处理失败: {document.file_name}, 错误: {str(e)}", exc_info=True)
            return ProcessResult.error_result(f"预处理时发生错误: {str(e)}", e)
    
    def _process_pdf(self, document: Document, image_utils: ImageUtils) -> List[str]:
        """
        处理PDF文件
        
        Args:
            document (Document): 文档对象
            image_utils (ImageUtils): 图像工具实例
            
        Returns:
            List[str]: 处理后的图像文件路径列表
        """
        try:
            # 提取文件名(不包含扩展名)
            base_name = os.path.splitext(document.file_name)[0]
            
            # 将PDF转换为图像
            logger.info(f"将PDF转换为图像: {document.file_path}")
            images = image_utils.pdf_to_images(
                document.file_path, 
                dpi=300, 
                output_format=self.config['output_format']
            )
            
            # 处理每一页图像
            processed_files = []
            for i, img in enumerate(images):
                # 组织输出文件路径
                output_path = os.path.join(
                    self.config['output_directory'],
                    f"{base_name}_page_{i+1}.{self.config['output_format'].lower()}"
                )
                
                # 处理图像并保存
                processed_img = self._enhance_image(img, image_utils)
                if processed_img:
                    image_utils.save_image(processed_img, output_path)
                    processed_files.append(output_path)
                    logger.debug(f"已处理并保存PDF第{i+1}页: {output_path}")
            
            return processed_files
            
        except Exception as e:
            logger.error(f"PDF处理失败: {document.file_path}, 错误: {str(e)}", exc_info=True)
            return []
    
    def _process_image(self, file_path: str, image_utils: ImageUtils) -> str:
        """
        处理单个图像文件
        
        Args:
            file_path (str): 图像文件路径
            image_utils (ImageUtils): 图像工具实例
            
        Returns:
            str: 处理后的图像文件路径
        """
        try:
            # 提取文件名(不包含扩展名)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # 组织输出文件路径
            output_path = os.path.join(
                self.config['output_directory'],
                f"{base_name}_processed.{self.config['output_format'].lower()}"
            )
            
            # 加载图像
            img = image_utils.load_image(file_path)
            
            # 增强图像
            processed_img = self._enhance_image(img, image_utils)
            
            # 保存处理后的图像
            if processed_img:
                image_utils.save_image(processed_img, output_path)
                logger.debug(f"已处理并保存图像: {output_path}")
                return output_path
            
            return ""
            
        except Exception as e:
            logger.error(f"图像处理失败: {file_path}, 错误: {str(e)}", exc_info=True)
            return ""
    
    def _enhance_image(self, image, image_utils: ImageUtils):
        """
        增强图像质量
        
        Args:
            image: PIL图像对象
            image_utils (ImageUtils): 图像工具实例
            
        Returns:
            PIL.Image: 增强后的图像对象
        """
        try:
            # 转换为灰度图（如果配置为True）
            if self.config.get('convert_to_grayscale', True):
                image = image_utils.convert_to_grayscale(image)
            
            # 应用图像增强
            image = image_utils.enhance_image(
                image,
                brightness=self.config['brightness'],
                contrast=self.config['contrast'],
                sharpness=self.config['sharpness']
            )
            
            # 去除噪点
            if self.config['noise_removal'] != 'none':
                image = image_utils.remove_noise(image, method=self.config['noise_removal'])
            
            # 倾斜校正
            if self.config['deskew_enabled']:
                image = image_utils.correct_skew(image)
            
            # 二值化处理
            if self.config['binarize'] != 'none':
                image = image_utils.binarize(image, method=self.config['binarize'])
            
            return image
            
        except Exception as e:
            logger.error(f"图像增强失败, 错误: {str(e)}", exc_info=True)
            return None
