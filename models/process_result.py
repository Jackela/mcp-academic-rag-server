"""
ProcessResult 类表示处理器处理文档后的结果。

该模块提供了ProcessResult类，用于存储处理是否成功、状态消息、
错误信息以及处理产生的数据。
"""

from typing import Dict, Optional, Any


class ProcessResult:
    """
    处理结果类表示处理器处理文档后的结果。
    
    该类存储处理是否成功、状态消息、错误信息以及处理产生的数据。
    它提供方法用于检查处理是否成功，以及获取处理状态信息和数据。
    """
    
    def __init__(self, success: bool, message: str, data: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None):
        """
        初始化ProcessResult对象。
        
        Args:
            success: 处理是否成功
            message: 处理状态消息
            data: 处理产生的数据，默认为None
            error: 处理过程中的异常，默认为None
        """
        self.success = success
        self.message = message
        self.data = data or {}
        self.error = error
    
    def is_successful(self) -> bool:
        """
        检查处理是否成功。
        
        Returns:
            如果处理成功则返回True，否则返回False
        """
        return self.success
    
    def get_message(self) -> str:
        """
        获取处理状态消息。
        
        Returns:
            处理状态消息
        """
        return self.message
    
    def get_data(self) -> Dict[str, Any]:
        """
        获取处理产生的数据。
        
        Returns:
            处理产生的数据字典
        """
        return self.data
    
    def get_error(self) -> Optional[Exception]:
        """
        获取处理过程中的异常。
        
        Returns:
            处理过程中的异常，如无异常则返回None
        """
        return self.error
    
    @classmethod
    def success_result(cls, message: str = "处理成功", data: Optional[Dict[str, Any]] = None) -> 'ProcessResult':
        """
        创建表示成功的ProcessResult对象。
        
        使用工厂方法模式简化成功结果的创建。
        
        Args:
            message: 成功消息，默认为"处理成功"
            data: 处理产生的数据，默认为None
            
        Returns:
            表示成功的ProcessResult对象
        """
        return cls(True, message, data)
    
    @classmethod
    def error_result(cls, message: str, error: Optional[Exception] = None) -> 'ProcessResult':
        """
        创建表示错误的ProcessResult对象。
        
        使用工厂方法模式简化错误结果的创建。
        
        Args:
            message: 错误消息
            error: 异常对象，默认为None
            
        Returns:
            表示错误的ProcessResult对象
        """
        return cls(False, message, error=error)
