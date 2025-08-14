"""
Haystack LLM 连接器模块 - 提供与Haystack框架集成的LLM接口
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, Callable
import json

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

# 配置日志
logger = logging.getLogger(__name__)

class HaystackLLMConnector:
    """Haystack LLM 连接器"""
    
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-3.5-turbo",
        api_base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        streaming_callback: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        初始化Haystack LLM连接器
        
        Args:
            api_key (str): OpenAI API密钥
            model (str): 模型名称，默认为"gpt-3.5-turbo"
            api_base_url (str): API基础URL，默认为"https://api.openai.com/v1"
            timeout (int): 请求超时时间(秒)，默认为60
            streaming_callback (callable, optional): 用于流式响应的回调函数
            parameters (dict, optional): LLM参数，如temperature、max_tokens等
        """
        self.api_key = api_key
        self.model = model
        self.api_base_url = api_base_url
        self.timeout = timeout
        self.streaming_callback = streaming_callback
        self.parameters = parameters or {}
        
        # 初始化OpenAI生成器
        self._init_generator()
    
    def _init_generator(self):
        """初始化OpenAI Chat生成器"""
        try:
            self.generator = OpenAIChatGenerator(
                api_key=Secret.from_token(self.api_key),
                model=self.model,
                api_base_url=self.api_base_url,
                timeout=self.timeout,
                streaming_callback=self.streaming_callback,
                generation_kwargs=self.parameters
            )
            logger.info(f"成功初始化Haystack OpenAI生成器: {self.model}")
        except Exception as e:
            logger.error(f"初始化Haystack OpenAI生成器失败: {str(e)}")
            raise
    
    def generate(
        self, 
        messages: List[Dict[str, str]],
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成对话响应
        
        Args:
            messages (list): 消息列表，格式为[{"role": "user", "content": "消息内容"}]
            generation_kwargs (dict, optional): 生成参数，覆盖默认参数
            
        Returns:
            dict: 生成结果
        """
        try:
            # 转换消息格式为Haystack的ChatMessage
            chat_messages = []
            for msg in messages:
                chat_messages.append(
                    ChatMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        name=msg.get("name")
                    )
                )
            
            # 合并生成参数
            params = self.parameters.copy()
            if generation_kwargs:
                params.update(generation_kwargs)
            
            # 设置临时生成参数
            temp_generator = self.generator
            if generation_kwargs:
                temp_generator.generation_kwargs = params
            
            # 生成响应
            logger.debug(f"开始生成对话响应: {json.dumps([m.dict() for m in chat_messages], ensure_ascii=False)[:200]}...")
            response = temp_generator.run(messages=chat_messages)
            
            # 提取结果
            result = {
                "content": response["replies"][0].content,
                "role": "assistant",
                "model": self.model
            }
            logger.debug(f"生成对话响应成功: {result['content'][:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"生成对话响应失败: {str(e)}")
            return {
                "content": f"生成对话响应失败: {str(e)}",
                "role": "assistant",
                "model": self.model,
                "error": str(e)
            }
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """
        更新LLM参数
        
        Args:
            parameters (dict): 新的LLM参数
        """
        self.parameters.update(parameters)
        self.generator.generation_kwargs = self.parameters
        logger.info(f"已更新LLM参数: {json.dumps(parameters, ensure_ascii=False)}")
    
    def set_model(self, model: str):
        """
        设置模型
        
        Args:
            model (str): 模型名称
        """
        if model != self.model:
            self.model = model
            self._init_generator()
            logger.info(f"已更新模型: {model}")


class HaystackLLMFactory:
    """Haystack LLM工厂类，创建Haystack LLM连接器"""
    
    @staticmethod
    def create_connector(
        config: Dict[str, Any],
        streaming_callback: Optional[Callable] = None
    ) -> HaystackLLMConnector:
        """
        创建Haystack LLM连接器
        
        Args:
            config (dict): LLM配置
            streaming_callback (callable, optional): 用于流式响应的回调函数
            
        Returns:
            HaystackLLMConnector: Haystack LLM连接器实例
        """
        api_key = config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
        model = config.get("model", "gpt-3.5-turbo")
        api_base_url = config.get("api_base_url", "https://api.openai.com/v1")
        timeout = config.get("timeout", 60)
        parameters = config.get("parameters", {})
        
        return HaystackLLMConnector(
            api_key=api_key,
            model=model,
            api_base_url=api_base_url,
            timeout=timeout,
            streaming_callback=streaming_callback,
            parameters=parameters
        )
