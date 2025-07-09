
"""
图像处理工具模块 - 提供学术文献OCR流水线所需的图像处理功能
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import logging

# 配置日志
logger = logging.getLogger(__name__)

class ImageUtils:
    """图像处理工具类，提供图像增强、校正和预处理功能"""
    
    @staticmethod
    def load_image(file_path):
        """
        加载图像文件
        
        Args:
            file_path (str): 图像文件路径
            
        Returns:
            PIL.Image: 加载的图像对象
            
        Raises:
            FileNotFoundError: 文件不存在时抛出
            ValueError: 不支持的文件格式时抛出
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"图像文件不存在: {file_path}")
            
            img = Image.open(file_path)
            logger.debug(f"成功加载图像: {file_path}, 尺寸: {img.size}, 格式: {img.format}")
            return img
        except Exception as e:
            logger.error(f"加载图像失败: {file_path}, 错误: {str(e)}")
            raise
    
    @staticmethod
    def save_image(image, output_path, format=None, quality=95):
        """
        保存图像到文件
        
        Args:
            image (PIL.Image): 图像对象
            output_path (str): 输出文件路径
            format (str, optional): 图像格式。若不指定，将从路径推断
            quality (int, optional): JPEG压缩质量 (1-100)
            
        Returns:
            bool: 保存成功返回True
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path, format=format, quality=quality)
            logger.debug(f"图像已保存至: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存图像失败: {output_path}, 错误: {str(e)}")
            return False
    
    @staticmethod
    def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
        """
        增强图像质量
        
        Args:
            image (PIL.Image): 输入图像
            brightness (float): 亮度调整因子(0.0-2.0)
            contrast (float): 对比度调整因子(0.0-2.0)
            sharpness (float): 锐度调整因子(0.0-2.0)
            
        Returns:
            PIL.Image: 增强后的图像
        """
        try:
            # 亮度增强
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
            
            # 对比度增强
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
            
            # 锐度增强
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(sharpness)
            
            logger.debug("图像增强完成")
            return image
        except Exception as e:
            logger.error(f"图像增强失败, 错误: {str(e)}")
            return image  # 出错时返回原始图像
    
    @staticmethod
    def remove_noise(image, method='median'):
        """
        去除图像噪点
        
        Args:
            image (PIL.Image): 输入图像
            method (str): 去噪方法，'median'或'gaussian'
            
        Returns:
            PIL.Image: 去噪后的图像
        """
        try:
            if method == 'median':
                return image.filter(ImageFilter.MedianFilter(size=3))
            elif method == 'gaussian':
                return image.filter(ImageFilter.GaussianBlur(radius=1))
            else:
                logger.warning(f"不支持的去噪方法: {method}，返回原始图像")
                return image
        except Exception as e:
            logger.error(f"图像去噪失败, 错误: {str(e)}")
            return image
    
    @staticmethod
    def correct_skew(image, delta=1, limit=5):
        """
        校正图像倾斜
        
        Args:
            image (PIL.Image): 输入图像
            delta (float): 角度增量
            limit (float): 最大校正角度
            
        Returns:
            PIL.Image: 校正后的图像，失败时返回原始图像
        """
        try:
            # 转换为OpenCV格式
            img_cv = np.array(image.convert('RGB'))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # 转灰度图
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            
            # 二值化
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # 寻找最佳角度
            angles = np.arange(-limit, limit + delta, delta)
            scores = []
            
            for angle in angles:
                # 旋转图像
                height, width = thresh.shape[:2]
                center = (width / 2, height / 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(thresh, rotation_matrix, (width, height), 
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
                # 计算水平投影的方差，方差越大说明文本行越明显
                projection = cv2.reduce(rotated, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
                scores.append(np.var(projection))
            
            # 找到最大方差对应的角度
            best_angle = angles[np.argmax(scores)]
            
            # 如果角度接近0，则不旋转
            if abs(best_angle) < 0.5:
                logger.debug("图像无需校正")
                return image
            
            # 旋转图像
            height, width = img_cv.shape[:2]
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            rotated = cv2.warpAffine(img_cv, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # 转回PIL格式
            rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(rotated_rgb)
            
            logger.debug(f"图像倾斜校正完成, 角度: {best_angle:.2f}°")
            return result
            
        except Exception as e:
            logger.error(f"图像倾斜校正失败, 错误: {str(e)}")
            return image  # 出错时返回原始图像
    
    @staticmethod
    def detect_orientation(image):
        """
        检测图像方向并自动旋转为正向
        适用于检测文档是否需要旋转90/180/270度
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 旋转后的图像
            int: 旋转角度
        """
        try:
            # 使用OCR引擎检测方向的代码应放这里
            # 由于需要OCR引擎支持，此处返回原始图像和0度
            logger.warning("图像方向检测功能需要OCR引擎支持")
            return image, 0
        except Exception as e:
            logger.error(f"图像方向检测失败, 错误: {str(e)}")
            return image, 0
    
    @staticmethod
    def binarize(image, method='adaptive'):
        """
        图像二值化处理，提高OCR文本识别率
        
        Args:
            image (PIL.Image): 输入图像
            method (str): 二值化方法，'otsu'或'adaptive'
            
        Returns:
            PIL.Image: 二值化后的图像
        """
        try:
            # 转为灰度图
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            
            # 转换为OpenCV格式
            img_cv = np.array(gray)
            
            if method == 'otsu':
                # Otsu二值化
                _, binary = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == 'adaptive':
                # 自适应二值化
                binary = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
            else:
                logger.warning(f"不支持的二值化方法: {method}，使用Otsu方法")
                _, binary = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 转回PIL格式
            result = Image.fromarray(binary)
            logger.debug(f"图像二值化完成, 方法: {method}")
            return result
            
        except Exception as e:
            logger.error(f"图像二值化失败, 错误: {str(e)}")
            return image
    
    @staticmethod
    def segment_page(image):
        """
        页面分割，识别文档中的文本区域、图像区域和表格区域
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            dict: 包含各区域坐标的字典
                {
                    'text_regions': [(x1,y1,x2,y2), ...],
                    'image_regions': [(x1,y1,x2,y2), ...],
                    'table_regions': [(x1,y1,x2,y2), ...]
                }
        """
        try:
            # 转换为OpenCV格式
            img_cv = np.array(image.convert('RGB'))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # 转灰度图
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # 简单区域分割示例 - 实际应用需要更复杂的算法
            # 此处仅做简单演示，返回整个页面为文本区域
            height, width = gray.shape
            regions = {
                'text_regions': [(0, 0, width, height)],
                'image_regions': [],
                'table_regions': []
            }
            
            logger.debug("页面区域分割完成")
            return regions
            
        except Exception as e:
            logger.error(f"页面区域分割失败, 错误: {str(e)}")
            # 返回整个页面为文本区域
            width, height = image.size
            return {
                'text_regions': [(0, 0, width, height)],
                'image_regions': [],
                'table_regions': []
            }
    
    @staticmethod
    def convert_to_grayscale(image):
        """
        将图像转换为灰度图
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 灰度图像
        """
        try:
            if image.mode != 'L':
                gray = image.convert('L')
                logger.debug("图像已转换为灰度")
                return gray
            return image
        except Exception as e:
            logger.error(f"转换灰度图失败, 错误: {str(e)}")
            return image
            
    @staticmethod
    def pdf_to_images(pdf_path, dpi=300, output_format='JPEG'):
        """
        将PDF文件转换为图像列表
        需要安装：pip install pdf2image poppler
        
        Args:
            pdf_path (str): PDF文件路径
            dpi (int): 输出图像的DPI
            output_format (str): 输出图像格式
            
        Returns:
            list: PIL.Image对象列表，每个元素对应PDF的一页
        """
        try:
            from pdf2image import convert_from_path
            
            logger.info(f"开始将PDF转换为图像: {pdf_path}")
            images = convert_from_path(pdf_path, dpi=dpi, fmt=output_format)
            logger.info(f"PDF转换完成，共{len(images)}页")
            return images
        except ImportError:
            logger.error("未安装pdf2image库，无法进行PDF转换")
            raise ImportError("请安装依赖: pip install pdf2image poppler")
        except Exception as e:
            logger.error(f"PDF转图像失败: {pdf_path}, 错误: {str(e)}")
            raise
            
    @staticmethod
    def crop_image(image, box):
        """
        裁剪图像指定区域
        
        Args:
            image (PIL.Image): 输入图像
            box (tuple): 裁剪区域坐标 (left, upper, right, lower)
            
        Returns:
            PIL.Image: 裁剪后的图像
        """
        try:
            cropped = image.crop(box)
            logger.debug(f"图像裁剪完成: {box}")
            return cropped
        except Exception as e:
            logger.error(f"图像裁剪失败, 错误: {str(e)}")
            return image
