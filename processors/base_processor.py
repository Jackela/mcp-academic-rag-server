"""
处理器接口和基类，定义所有文档处理器必须实现的方法。

该模块提供了IProcessor接口和BaseProcessor基类，用于确保所有处理器
遵循统一的接口规范，提高系统的一致性和可扩展性。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from models.document import Document
from models.process_result import ProcessResult


class IProcessor(ABC):
    """
    处理器接口，定义所有文档处理器必须实现的方法。
    
    所有处理器类必须继承此接口并实现其方法。这确保了系统中所有处理器
    都具有一致的接口，便于在Pipeline中进行组合和替换。
    """
    
    @abstractmethod
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档并返回处理结果。
        
        Args:
            document: 要处理的Document对象
            
        Returns:
            表示处理结果的ProcessResult对象
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        获取处理器名称。
        
        Returns:
            处理器名称
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        获取处理器描述。
        
        Returns:
            处理器描述
        """
        pass
    
    def get_stage(self) -> str:
        """
        获取处理器对应的处理阶段。
        默认使用处理器名称作为阶段名称。
        
        Returns:
            处理阶段名称
        """
        return self.get_name()
    
    def supports_file_type(self, file_type: str) -> bool:
        """
        检查处理器是否支持特定文件类型。
        默认支持所有文件类型，子类可以重写此方法实现文件类型过滤。
        
        Args:
            file_type: 文件类型（扩展名，如.pdf、.jpg等）
            
        Returns:
            如果支持该文件类型则返回True，否则返回False
        """
        return True
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        设置处理器配置。
        
        Args:
            config: 配置字典
        """
        self.config = config
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取处理器配置。
        
        Returns:
            配置字典
        """
        return getattr(self, 'config', {})


class BaseProcessor(IProcessor):
    """
    处理器基类，提供IProcessor接口的基本实现。
    
    具体处理器可以继承此类，以获得一些通用功能的实现，减少重复代码。
    该类实现了接口中的大部分方法，只保留process方法为抽象方法，需要子类实现。
    """
    
    def __init__(self, name: str = None, description: str = None, config: Dict[str, Any] = None):
        """
        初始化BaseProcessor对象。
        
        Args:
            name: 处理器名称，默认为类名
            description: 处理器描述，默认为空字符串
            config: 处理器配置，默认为空字典
        """
        self._name = name or self.__class__.__name__
        self._description = description or ""
        self.config = config or {}
    
    def get_name(self) -> str:
        """
        获取处理器名称。
        
        Returns:
            处理器名称
        """
        return self._name
    
    def get_description(self) -> str:
        """
        获取处理器描述。
        
        Returns:
            处理器描述
        """
        return self._description
    
    @abstractmethod
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档并返回处理结果。
        
        Args:
            document: 要处理的Document对象
            
        Returns:
            表示处理结果的ProcessResult对象
        """
        pass
