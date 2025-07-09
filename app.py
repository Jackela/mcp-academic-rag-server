"""
学术文献OCR电子化与智能检索系统主入口。

该模块是系统的入口点，负责初始化系统、加载配置、创建处理流水线，
并提供命令行接口用于处理文档和执行查询。
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

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


def load_processors(config: Dict[str, Any]) -> Dict[str, IProcessor]:
    """
    根据配置动态加载处理器。
    
    使用工厂模式创建处理器实例，支持根据配置启用或禁用处理器。
    
    Args:
        config: 处理器配置字典
        
    Returns:
        处理器字典，键为处理器名称，值为处理器实例
    """
    processors = {}
    
    # 在实际实现中，这里应该使用反射或导入模块的方式动态创建处理器实例
    # 由于处理器类尚未实现，此处返回空字典
    # 
    # 实际实现示例:
    # for processor_name, processor_config in config.items():
    #     if processor_config.get("enabled", True):
    #         try:
    #             # 导入处理器类
    #             module_name = f"processors.{processor_name.lower()}"
    #             module = importlib.import_module(module_name)
    #             processor_class = getattr(module, processor_name)
    #             
    #             # 创建处理器实例并设置配置
    #             processor = processor_class()
    #             processor.set_config(processor_config)
    #             processors[processor_name] = processor
    #         except Exception as e:
    #             logging.error(f"加载处理器 {processor_name} 失败: {str(e)}")
    
    return processors


def create_pipeline(processors: Dict[str, IProcessor], name: str = "MainPipeline") -> Pipeline:
    """
    创建处理流水线。
    
    根据预定义的顺序组装处理器，形成完整的处理流水线。
    
    Args:
        processors: 处理器字典
        name: 流水线名称
        
    Returns:
        创建的Pipeline实例
    """
    pipeline = Pipeline(name)
    
    # 按特定顺序添加处理器
    processor_order = [
        "PreProcessor",
        "OCRProcessor",
        "StructureProcessor",
        "ClassificationProcessor",
        "FormatConverter",
        "EmbeddingProcessor"
    ]
    
    for processor_name in processor_order:
        if processor_name in processors:
            pipeline.add_processor(processors[processor_name])
    
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
