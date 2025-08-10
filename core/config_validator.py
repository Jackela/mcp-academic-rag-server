"""
配置验证模块

该模块提供配置文件结构验证、参数合法性检查和配置一致性验证功能。
"""

import logging
import os
from typing import Dict, Any, List, Optional, Set, Union
import json

# Optional jsonschema dependency for advanced validation
try:
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    ValidationError = Exception

logger = logging.getLogger(__name__)


class ConfigValidator:
    """配置验证器类"""
    
    # 定义配置模式
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "storage": {
                "type": "object",
                "properties": {
                    "base_path": {"type": "string"},
                    "output_path": {"type": "string"}
                },
                "required": ["base_path", "output_path"]
            },
            "processors": {
                "type": "object",
                "patternProperties": {
                    "^[a-zA-Z_][a-zA-Z0-9_]*$": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "config": {"type": "object"},
                            "module": {"type": "string"},
                            "class": {"type": "string"}
                        },
                        "required": ["enabled"]
                    }
                }
            },
            "connectors": {
                "type": "object"
            },
            "rag_settings": {
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                    "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "prompt_template": {"type": "string"}
                }
            },
            "llm": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "api_url": {"type": "string"},
                    "api_key": {"type": "string"},
                    "model": {"type": "string"},
                    "embedding_model": {"type": "string"},
                    "settings": {"type": "object"}
                }
            },
            "logging": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "format": {"type": "string"},
                    "file": {"type": "string"}
                }
            }
        },
        "required": ["storage", "processors"]
    }
    
    # 标准处理器名称映射（支持向后兼容）
    PROCESSOR_NAME_MAPPING = {
        # 旧名称 -> 新名称
        "PreProcessor": "pre_processor",
        "OCRProcessor": "ocr_processor", 
        "StructureProcessor": "structure_processor",
        "ClassificationProcessor": "classification_processor",
        "FormatConverter": "format_converter",
        "EmbeddingProcessor": "embedding_processor",
        # 新名称保持不变
        "pre_processor": "pre_processor",
        "ocr_processor": "ocr_processor",
        "structure_processor": "structure_processor", 
        "classification_processor": "classification_processor",
        "format_converter": "format_converter",
        "embedding_processor": "embedding_processor"
    }
    
    # 必需的处理器（按执行顺序）
    REQUIRED_PROCESSORS = [
        "pre_processor",
        "ocr_processor", 
        "structure_processor",
        "embedding_processor"
    ]
    
    def __init__(self):
        """初始化配置验证器"""
        self.validation_errors = []
        self.warnings = []
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置的完整性和合法性
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 验证是否通过
        """
        self.validation_errors = []
        self.warnings = []
        
        try:
            # JSON Schema验证（如果可用）
            if HAS_JSONSCHEMA:
                validate(instance=config, schema=self.CONFIG_SCHEMA)
                logger.info("配置结构验证通过")
            else:
                logger.warning("jsonschema未安装，跳过高级结构验证")
            
        except ValidationError as e:
            self.validation_errors.append(f"配置结构验证失败: {e.message}")
            logger.error(f"配置结构验证失败: {e.message}")
            return False
        except Exception as e:
            logger.warning(f"配置结构验证时出现警告: {str(e)}")
        
        # 自定义验证
        self._validate_storage_paths(config.get("storage", {}))
        self._validate_processors(config.get("processors", {}))
        self._validate_connectors(config.get("connectors", {}))
        self._validate_rag_settings(config.get("rag_settings", {}))
        self._validate_logging_config(config.get("logging", {}))
        
        # 输出警告
        for warning in self.warnings:
            logger.warning(warning)
        
        # 检查是否有验证错误
        if self.validation_errors:
            for error in self.validation_errors:
                logger.error(error)
            return False
        
        logger.info("配置验证完成，所有检查通过")
        return True
    
    def _validate_storage_paths(self, storage_config: Dict[str, Any]) -> None:
        """验证存储路径配置"""
        base_path = storage_config.get("base_path")
        output_path = storage_config.get("output_path")
        
        if base_path and not os.path.isabs(base_path):
            self.warnings.append(f"建议使用绝对路径作为base_path: {base_path}")
        
        if output_path and not os.path.isabs(output_path):
            self.warnings.append(f"建议使用绝对路径作为output_path: {output_path}")
    
    def _validate_processors(self, processors_config: Dict[str, Any]) -> None:
        """验证处理器配置"""
        if not processors_config:
            self.validation_errors.append("处理器配置不能为空")
            return
        
        # 标准化处理器名称
        normalized_processors = set()
        for processor_name in processors_config.keys():
            if processor_name in self.PROCESSOR_NAME_MAPPING:
                normalized_name = self.PROCESSOR_NAME_MAPPING[processor_name]
                normalized_processors.add(normalized_name)
            else:
                self.warnings.append(f"未知的处理器类型: {processor_name}")
        
        # 检查必需的处理器
        missing_processors = set(self.REQUIRED_PROCESSORS) - normalized_processors
        if missing_processors:
            self.validation_errors.append(f"缺少必需的处理器: {', '.join(missing_processors)}")
        
        # 检查处理器启用状态
        enabled_processors = []
        for processor_name, processor_config in processors_config.items():
            if processor_config.get("enabled", False):
                normalized_name = self.PROCESSOR_NAME_MAPPING.get(processor_name, processor_name)
                enabled_processors.append(normalized_name)
        
        if not enabled_processors:
            self.validation_errors.append("至少需要启用一个处理器")
        
        # 验证处理器顺序（基本检查）
        required_enabled = [p for p in self.REQUIRED_PROCESSORS if p in enabled_processors]
        if len(required_enabled) > 1:
            logger.info(f"启用的必需处理器: {', '.join(required_enabled)}")
    
    def _validate_connectors(self, connectors_config: Dict[str, Any]) -> None:
        """验证连接器配置"""
        if not connectors_config:
            self.warnings.append("未配置任何连接器")
            return
        
        # 检查OCR连接器
        ocr_connector = connectors_config.get("OCRAPIConnector", {})
        if ocr_connector:
            for api_name, api_config in ocr_connector.items():
                if not api_config.get("api_key"):
                    self.warnings.append(f"OCR连接器 {api_name} 缺少API密钥")
        
        # 检查LLM连接器
        llm_connector = connectors_config.get("LLMConnector", {})
        if llm_connector:
            if not llm_connector.get("api_key"):
                self.warnings.append("LLM连接器缺少API密钥")
            if not llm_connector.get("model"):
                self.warnings.append("LLM连接器缺少模型配置")
    
    def _validate_rag_settings(self, rag_config: Dict[str, Any]) -> None:
        """验证RAG设置"""
        if not rag_config:
            self.warnings.append("未配置RAG设置，将使用默认值")
            return
        
        top_k = rag_config.get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            self.validation_errors.append("top_k必须是1-100之间的整数")
        
        threshold = rag_config.get("threshold", 0.75)
        if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
            self.validation_errors.append("threshold必须是0.0-1.0之间的数值")
        
        prompt_template = rag_config.get("prompt_template")
        if prompt_template and not isinstance(prompt_template, str):
            self.validation_errors.append("prompt_template必须是字符串")
        elif prompt_template and "{context}" not in prompt_template:
            self.warnings.append("prompt_template建议包含{context}占位符")
    
    def _validate_logging_config(self, logging_config: Dict[str, Any]) -> None:
        """验证日志配置"""
        if not logging_config:
            self.warnings.append("未配置日志设置，将使用默认配置")
            return
        
        log_level = logging_config.get("level", "INFO")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_levels:
            self.validation_errors.append(f"日志级别必须是: {', '.join(valid_levels)}")
        
        log_file = logging_config.get("file")
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                self.warnings.append(f"日志目录不存在，需要创建: {log_dir}")
    
    def normalize_processor_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化处理器配置名称
        
        Args:
            config: 原始配置
            
        Returns:
            Dict[str, Any]: 标准化后的配置
        """
        if "processors" not in config:
            return config
        
        normalized_config = config.copy()
        old_processors = normalized_config["processors"]
        new_processors = {}
        
        for old_name, processor_config in old_processors.items():
            # 获取标准化名称
            new_name = self.PROCESSOR_NAME_MAPPING.get(old_name, old_name)
            
            # 如果是旧名称，记录转换
            if old_name != new_name:
                logger.info(f"处理器名称标准化: {old_name} -> {new_name}")
            
            new_processors[new_name] = processor_config
        
        normalized_config["processors"] = new_processors
        return normalized_config
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        获取验证报告
        
        Returns:
            Dict[str, Any]: 包含错误和警告的报告
        """
        return {
            "errors": self.validation_errors,
            "warnings": self.warnings,
            "is_valid": len(self.validation_errors) == 0
        }


def validate_config_file(config_path: str) -> tuple[bool, Dict[str, Any]]:
    """
    验证配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        tuple: (是否验证通过, 验证报告)
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        validator = ConfigValidator()
        is_valid = validator.validate_config(config)
        report = validator.get_validation_report()
        
        return is_valid, report
        
    except FileNotFoundError:
        return False, {"errors": [f"配置文件不存在: {config_path}"], "warnings": []}
    except json.JSONDecodeError as e:
        return False, {"errors": [f"配置文件JSON格式错误: {str(e)}"], "warnings": []}
    except Exception as e:
        return False, {"errors": [f"验证配置文件时发生错误: {str(e)}"], "warnings": []}


# 默认配置生成器
def generate_default_config() -> Dict[str, Any]:
    """
    生成默认配置
    
    Returns:
        Dict[str, Any]: 默认配置字典
    """
    return {
        "storage": {
            "base_path": "./data",
            "output_path": "./output"
        },
        "processors": {
            "pre_processor": {
                "enabled": True,
                "config": {
                    "enhance_image": True,
                    "correct_skew": True
                }
            },
            "ocr_processor": {
                "enabled": True,
                "config": {
                    "api": "azure",
                    "language": "zh-Hans"
                }
            },
            "structure_processor": {
                "enabled": True,
                "config": {
                    "detect_sections": True,
                    "detect_headings": True
                }
            },
            "classification_processor": {
                "enabled": False,
                "config": {
                    "api": "openai",
                    "min_confidence": 0.7
                }
            },
            "format_converter": {
                "enabled": False,
                "config": {
                    "output_formats": ["markdown"],
                    "preserve_layout": True
                }
            },
            "embedding_processor": {
                "enabled": True,
                "config": {
                    "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "batch_size": 32
                }
            }
        },
        "connectors": {
            "OCRAPIConnector": {
                "azure": {
                    "endpoint": "YOUR_AZURE_ENDPOINT",
                    "api_key": "YOUR_AZURE_API_KEY",
                    "region": "eastus"
                }
            },
            "LLMConnector": {
                "type": "openai",
                "api_url": "https://api.openai.com/v1",
                "api_key": "YOUR_OPENAI_API_KEY",
                "model": "gpt-3.5-turbo",
                "embedding_model": "text-embedding-3-small",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            }
        },
        "rag_settings": {
            "top_k": 5,
            "threshold": 0.75,
            "prompt_template": "基于以下文献内容回答问题:\n\n{context}\n\n问题: {question}\n\n回答:"
        },
        "llm": {
            "type": "openai",
            "api_url": "https://api.openai.com/v1",
            "api_key": "YOUR_OPENAI_API_KEY",
            "model": "gpt-3.5-turbo",
            "embedding_model": "text-embedding-3-small",
            "settings": {
                "temperature": 0.7,
                "max_tokens": 2000
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "./logs/system.log"
        }
    }