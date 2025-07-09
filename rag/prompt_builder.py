"""
提示构建器模块 - 实现基于Haystack的提示模板构建器
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable
import json

from haystack import component
from haystack.dataclasses import ChatMessage, Document

# 配置日志
logger = logging.getLogger(__name__)

@component
class ChatPromptBuilder:
    """基于Haystack的聊天提示构建器"""
    
    def __init__(
        self,
        system_prompt: str = None,
        document_separator: str = "\n---\n",
        include_citation: bool = True,
        max_context_length: int = 4000,
        template_type: str = "academic"
    ):
        """
        初始化聊天提示构建器
        
        Args:
            system_prompt (str, optional): 系统提示
            document_separator (str): 文档分隔符
            include_citation (bool): 是否包含引用信息
            max_context_length (int): 最大上下文长度
            template_type (str): 模板类型，可选值："academic"、"general"、"concise"
        """
        self.document_separator = document_separator
        self.include_citation = include_citation
        self.max_context_length = max_context_length
        self.template_type = template_type
        
        # 设置系统提示
        self.system_prompt = system_prompt or self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示"""
        if self.template_type == "academic":
            return (
                "你是一个学术研究助手，可以帮助回答与学术文献相关的问题。"
                "根据提供的学术文档上下文回答用户问题。"
                "注意严格基于上下文信息回答问题，避免编造信息。"
                "如果上下文中没有足够信息，请诚实说明。"
                "如果引用信息，请注明引用来源（文档ID和标题）。"
                "回答应该客观、准确、全面，使用学术风格，避免过度简化。"
            )
        elif self.template_type == "concise":
            return (
                "你是一个简洁的学术助手，提供精确且简明的回答。"
                "基于提供的学术文档回答问题，避免冗长解释。"
                "只使用上下文中的信息，不添加额外内容。"
                "如需引用，提供简短引用标记。"
            )
        else:  # general
            return (
                "你是一个智能助手，负责回答用户关于文档内容的问题。"
                "根据提供的文档上下文回答用户问题。"
                "回答应基于上下文信息，避免编造事实。"
                "如果上下文信息不足，请坦诚告知用户。"
                "保持回答清晰、有条理，易于理解。"
            )
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        格式化文档列表为上下文字符串
        
        Args:
            documents (List[Document]): 文档列表
            
        Returns:
            str: 格式化后的上下文
        """
        formatted_docs = []
        total_length = 0
        
        for doc in documents:
            # 获取文档内容
            content = doc.content
            
            # 获取文档元数据
            metadata = doc.metadata or {}
            title = metadata.get("title", "未知标题")
            
            # 格式化文档
            if self.include_citation:
                doc_str = f"文档ID: {doc.id}\n标题: {title}\n内容:\n{content}"
            else:
                doc_str = content
            
            # 检查长度是否超出限制
            if total_length + len(doc_str) + len(self.document_separator) > self.max_context_length:
                # 如果添加整个文档会超出限制，考虑添加部分内容
                remaining_length = self.max_context_length - total_length - len(self.document_separator)
                if remaining_length > 100:  # 至少保留100个字符才有意义
                    if self.include_citation:
                        citation_part = f"文档ID: {doc.id}\n标题: {title}\n内容:\n"
                        content_part = content[:remaining_length - len(citation_part)]
                        doc_str = citation_part + content_part + "...(内容被截断)"
                    else:
                        doc_str = content[:remaining_length] + "...(内容被截断)"
                    
                    formatted_docs.append(doc_str)
                
                break
            
            formatted_docs.append(doc_str)
            total_length += len(doc_str) + len(self.document_separator)
        
        # 使用分隔符连接文档
        context = self.document_separator.join(formatted_docs)
        
        logger.debug(f"格式化了{len(formatted_docs)}个文档，总长度: {len(context)}")
        return context
    
    def _build_academic_prompt(self, query: str, context: str) -> str:
        """
        构建学术提示
        
        Args:
            query (str): 查询
            context (str): 上下文
            
        Returns:
            str: 构建的提示
        """
        return (
            f"请回答以下学术问题，并基于提供的学术文献内容。\n\n"
            f"问题：{query}\n\n"
            f"参考文献内容：\n{context}\n\n"
            f"回答要求：\n"
            f"1. 仅基于提供的参考文献内容回答\n"
            f"2. 使用学术语言，清晰准确地回答问题\n"
            f"3. 适当引用相关文献内容（使用文档ID和标题）\n"
            f"4. 如果参考文献不包含足够信息，请诚实说明\n"
            f"5. 避免添加未在参考文献中提及的信息\n"
            f"请基于以上要求，提供你的回答："
        )
    
    def _build_concise_prompt(self, query: str, context: str) -> str:
        """
        构建简洁提示
        
        Args:
            query (str): 查询
            context (str): 上下文
            
        Returns:
            str: 构建的提示
        """
        return (
            f"基于以下文献内容，简明扼要地回答问题。\n\n"
            f"问题：{query}\n\n"
            f"参考内容：\n{context}\n\n"
            f"要求：回答简洁准确，仅使用参考内容，必要时提供简短引用。"
        )
    
    def _build_general_prompt(self, query: str, context: str) -> str:
        """
        构建通用提示
        
        Args:
            query (str): 查询
            context (str): 上下文
            
        Returns:
            str: 构建的提示
        """
        return (
            f"请根据以下文档内容回答问题：\n\n"
            f"问题：{query}\n\n"
            f"参考文档：\n{context}\n\n"
            f"请基于参考文档提供准确、有帮助的回答。如果文档中没有足够信息，请坦诚说明。"
        )
    
    @component.output_types(messages=List[ChatMessage])
    def __call__(
        self,
        query: str,
        documents: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        构建提示
        
        Args:
            query (str): 用户查询
            documents (List[Document]): 检索到的文档列表
            chat_history (List[Dict], optional): 聊天历史记录
            
        Returns:
            Dict[str, List[ChatMessage]]: 包含消息列表的字典
        """
        try:
            # 格式化文档
            context = self._format_documents(documents)
            
            # 根据模板类型选择不同的提示构建方法
            if self.template_type == "academic":
                user_prompt = self._build_academic_prompt(query, context)
            elif self.template_type == "concise":
                user_prompt = self._build_concise_prompt(query, context)
            else:  # general
                user_prompt = self._build_general_prompt(query, context)
            
            # 创建消息列表
            messages = [ChatMessage(role="system", content=self.system_prompt)]
            
            # 添加聊天历史
            if chat_history:
                for message in chat_history:
                    messages.append(
                        ChatMessage(
                            role=message.get("role", "user"),
                            content=message.get("content", ""),
                            name=message.get("name")
                        )
                    )
            
            # 添加用户查询
            messages.append(ChatMessage(role="user", content=user_prompt))
            
            logger.debug(f"构建了提示，消息数量: {len(messages)}")
            return {"messages": messages}
            
        except Exception as e:
            logger.error(f"构建提示失败: {str(e)}")
            
            # 创建简单的错误消息
            error_message = f"处理查询时出错: {str(e)}"
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=query),
                ChatMessage(role="assistant", content=error_message)
            ]
            
            return {"messages": messages}
    
    def set_system_prompt(self, system_prompt: str):
        """
        设置系统提示
        
        Args:
            system_prompt (str): 新的系统提示
        """
        self.system_prompt = system_prompt
        logger.info("已更新系统提示")
    
    def set_template_type(self, template_type: str):
        """
        设置模板类型
        
        Args:
            template_type (str): 新的模板类型
        """
        self.template_type = template_type
        # 更新系统提示
        if not self.system_prompt or self.system_prompt == self._get_default_system_prompt():
            self.system_prompt = self._get_default_system_prompt()
        logger.info(f"已更新模板类型: {template_type}")


class PromptBuilderFactory:
    """提示构建器工厂类，创建不同类型的提示构建器"""
    
    @staticmethod
    def create_builder(
        template_type: str = "academic",
        system_prompt: str = None,
        include_citation: bool = True,
        max_context_length: int = 4000,
        config: Optional[Dict[str, Any]] = None
    ) -> ChatPromptBuilder:
        """
        创建提示构建器
        
        Args:
            template_type (str): 模板类型
            system_prompt (str, optional): 系统提示
            include_citation (bool): 是否包含引用
            max_context_length (int): 最大上下文长度
            config (dict, optional): 配置参数
            
        Returns:
            ChatPromptBuilder: 提示构建器实例
        """
        # 处理配置参数
        if config:
            template_type = config.get("template_type", template_type)
            system_prompt = config.get("system_prompt", system_prompt)
            include_citation = config.get("include_citation", include_citation)
            max_context_length = config.get("max_context_length", max_context_length)
        
        return ChatPromptBuilder(
            system_prompt=system_prompt,
            include_citation=include_citation,
            max_context_length=max_context_length,
            template_type=template_type
        )
