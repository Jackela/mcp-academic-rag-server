"""
学术文献OCR电子化与智能检索系统主入口。

该模块是系统的入口点，负责初始化系统、加载配置、创建处理流水线，
并提供命令行接口用于处理文档和执行查询。
"""

import os
import sys
import logging
import argparse
import importlib
from typing import Dict, Any, List

from core.config_manager import ConfigManager
from core.pipeline import Pipeline
from models.document import Document
from processors.base_processor import IProcessor


def setup_logging(config: Dict[str, Any]) -> None:
    """
    设置日志系统。
    
    Args:
        config: 日志配置字典
    """
    log_level = getattr(logging, config.get("level", "INFO"))
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("file")
    
    # 创建日志目录
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8') if log_file else logging.NullHandler(),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_processors(config: Dict[str, Any]) -> List[IProcessor]:
    """
    根据配置动态加载处理器。
    
    使用工厂模式创建处理器实例，支持根据配置启用或禁用处理器。
    
    Args:
        config: 处理器配置字典
        
    Returns:
        处理器列表，按配置顺序排列
    """
    processors = []
    logger = logging.getLogger('load_processors')
    
    # 默认处理器配置和映射
    default_processor_mapping = {
        'pre_processor': {
            'module': 'processors.pre_processor',
            'class': 'PreProcessor'
        },
        'ocr_processor': {
            'module': 'processors.ocr_processor', 
            'class': 'OCRProcessor'
        },
        'structure_processor': {
            'module': 'processors.structure_processor',
            'class': 'StructureProcessor'
        },
        'classification_processor': {
            'module': 'processors.classification_processor',
            'class': 'ClassificationProcessor'
        },
        'format_converter': {
            'module': 'processors.format_converter',
            'class': 'FormatConverterProcessor'
        },
        'embedding_processor': {
            'module': 'processors.haystack_embedding_processor',
            'class': 'HaystackEmbeddingProcessor'
        }
    }
    
    for processor_name, processor_config in config.items():
        if not processor_config.get("enabled", True):
            logger.info(f"跳过禁用的处理器: {processor_name}")
            continue
            
        try:
            # 获取模块和类名
            if processor_name in default_processor_mapping:
                # 使用默认映射
                mapping = default_processor_mapping[processor_name]
                module_path = mapping['module']
                class_name = mapping['class']
            else:
                # 使用配置中的映射
                module_path = processor_config.get("module", f"processors.{processor_name}")
                class_name = processor_config.get("class", f"{processor_name.title()}Processor")
            
            # 导入模块
            logger.info(f"正在加载处理器: {processor_name} 从 {module_path}.{class_name}")
            module = importlib.import_module(module_path)
            processor_class = getattr(module, class_name)
            
            # 创建处理器实例
            processor_init_config = processor_config.get("config", {})
            processor = processor_class(config=processor_init_config)
            processors.append(processor)
            
            logger.info(f"成功加载处理器: {processor_name}")
            
        except Exception as e:
            logger.error(f"加载处理器 {processor_name} 失败: {str(e)}")
            # 继续加载其他处理器
    
    logger.info(f"成功加载 {len(processors)} 个处理器")
    return processors


def create_pipeline(processors: List[IProcessor], name: str = "MainPipeline") -> Pipeline:
    """
    创建处理流水线。
    
    将加载的处理器按顺序添加到流水线中。
    
    Args:
        processors: 处理器列表
        name: 流水线名称
        
    Returns:
        创建的Pipeline实例
    """
    pipeline = Pipeline(name)
    logger = logging.getLogger('create_pipeline')
    
    # 按顺序添加处理器
    for processor in processors:
        pipeline.add_processor(processor)
        logger.info(f"添加处理器到流水线: {processor.get_name()}")
    
    logger.info(f"创建流水线 '{name}' 完成，包含 {len(processors)} 个处理器")
    return pipeline


def main():
    """
    应用主入口。
    
    处理命令行参数，初始化系统，执行文档处理或查询操作。
    """
    parser = argparse.ArgumentParser(description="学术文献OCR电子化与智能检索系统")
    parser.add_argument("--config", default="./config/config.json", help="配置文件路径")
    parser.add_argument("--input", help="输入文档路径")
    parser.add_argument("--query", help="查询文本")
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager(args.config)
    
    # 设置日志
    setup_logging(config_manager.get_value("logging", {}))
    
    logger = logging.getLogger("app")
    logger.info("学术文献OCR电子化与智能检索系统启动")
    
    # 创建存储目录
    storage_base_path = config_manager.get_value("storage.base_path", "./data")
    storage_output_path = config_manager.get_value("storage.output_path", "./output")
    
    os.makedirs(storage_base_path, exist_ok=True)
    os.makedirs(storage_output_path, exist_ok=True)
    
    # 加载处理器
    processors = load_processors(config_manager.get_value("processors", {}))
    
    # 创建处理流水线
    pipeline = create_pipeline(processors)
    
    # 处理输入文档
    if args.input:
        if os.path.exists(args.input):
            logger.info(f"处理文档: {args.input}")
            document = Document(args.input)
            result = pipeline.process_document(document)
            
            if result.is_successful():
                logger.info(f"文档处理成功: {result.get_message()}")
                # 这里可以输出处理结果的摘要
            else:
                logger.error(f"文档处理失败: {result.get_message()}")
        else:
            logger.error(f"输入文档不存在: {args.input}")
    
    # 处理查询
    elif args.query:
        logger.info(f"处理查询: {args.query}")
        # 这里应该实现RAG逻辑，处理自然语言查询
        # 由于RAG部分尚未实现，此处略过
        logger.warning("查询功能尚未实现")
    
    # 如果没有输入参数，打印使用说明
    else:
        parser.print_help()
    
    logger.info("系统执行完毕")


if __name__ == "__main__":
    main()
