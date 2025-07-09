"""
聊天会话模块 - 实现基于Haystack的聊天会话管理
"""

import os
import logging
import json
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime

from haystack.dataclasses import Document

from .haystack_pipeline import RAGPipeline

# 配置日志
logger = logging.getLogger(__name__)

class Message:
    """聊天消息类"""
    
    def __init__(
        self,
        role: str,
        content: str,
        message_id: str = None,
        timestamp: float = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化聊天消息
        
        Args:
            role (str): 消息角色，如"user"、"assistant"、"system"
            content (str): 消息内容
            message_id (str, optional): 消息ID，如果为None则自动生成
            timestamp (float, optional): 消息时间戳，如果为None则使用当前时间
            metadata (dict, optional): 消息元数据
        """
        self.role = role
        self.content = content
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now().timestamp()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            dict: 消息字典
        """
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        从字典创建消息
        
        Args:
            data (dict): 消息字典
            
        Returns:
            Message: 消息实例
        """
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            message_id=data.get("message_id"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )


class Citation:
    """引用类"""
    
    def __init__(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化引用
        
        Args:
            document_id (str): 文档ID
            text (str): 引用文本
            metadata (dict, optional): 引用元数据
        """
        self.document_id = document_id
        self.text = text
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            dict: 引用字典
        """
        return {
            "document_id": self.document_id,
            "text": self.text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Citation':
        """
        从字典创建引用
        
        Args:
            data (dict): 引用字典
            
        Returns:
            Citation: 引用实例
        """
        return cls(
            document_id=data.get("document_id", ""),
            text=data.get("text", ""),
            metadata=data.get("metadata", {})
        )


class ChatSession:
    """聊天会话类"""
    
    def __init__(
        self,
        session_id: str = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        max_history_length: int = 10,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化聊天会话
        
        Args:
            session_id (str, optional): 会话ID，如果为None则自动生成
            rag_pipeline (RAGPipeline, optional): RAG管道
            max_history_length (int): 最大历史记录长度
            metadata (dict, optional): 会话元数据
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.rag_pipeline = rag_pipeline
        self.max_history_length = max_history_length
        self.metadata = metadata or {}
        self.messages = []
        self.citations = {}  # 消息ID到引用列表的映射
        self.created_at = datetime.now().timestamp()
        self.last_active_at = self.created_at
    
    def add_message(
        self,
        role: str,
        content: str,
        message_id: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        添加消息
        
        Args:
            role (str): 消息角色
            content (str): 消息内容
            message_id (str, optional): 消息ID
            metadata (dict, optional): 消息元数据
            
        Returns:
            Message: 添加的消息
        """
        # 创建消息
        message = Message(
            role=role,
            content=content,
            message_id=message_id,
            metadata=metadata
        )
        
        # 添加到消息列表
        self.messages.append(message)
        
        # 限制历史记录长度
        if len(self.messages) > self.max_history_length * 2:  # 乘以2是因为用户和助手的消息成对出现
            # 保留最近的消息
            self.messages = self.messages[-self.max_history_length * 2:]
        
        # 更新最后活动时间
        self.last_active_at = datetime.now().timestamp()
        
        logger.debug(f"会话{self.session_id}添加了{role}消息: {content[:50]}...")
        return message
    
    def add_citation(
        self,
        message_id: str,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Citation:
        """
        添加引用
        
        Args:
            message_id (str): 消息ID
            document_id (str): 文档ID
            text (str): 引用文本
            metadata (dict, optional): 引用元数据
            
        Returns:
            Citation: 添加的引用
        """
        # 创建引用
        citation = Citation(
            document_id=document_id,
            text=text,
            metadata=metadata
        )
        
        # 添加到引用映射
        if message_id not in self.citations:
            self.citations[message_id] = []
        
        self.citations[message_id].append(citation)
        
        logger.debug(f"会话{self.session_id}为消息{message_id}添加了引用: 文档{document_id}")
        return citation
    
    def process_query(
        self,
        query: str,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[Message, List[Document]]:
        """
        处理查询
        
        Args:
            query (str): 用户查询
            generation_kwargs (dict, optional): 生成参数
            
        Returns:
            tuple: (回答消息, 检索到的文档列表)
            
        Raises:
            ValueError: 如果RAG管道未设置
        """
        if not self.rag_pipeline:
            raise ValueError("RAG管道未设置，无法处理查询")
        
        # 添加用户消息
        user_message = self.add_message(role="user", content=query)
        
        # 获取聊天历史
        chat_history = []
        for msg in self.messages[:-1]:  # 排除刚刚添加的用户消息
            chat_history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        try:
            # 运行RAG管道
            result = self.rag_pipeline.run(
                query=query,
                chat_history=chat_history,
                generation_kwargs=generation_kwargs
            )
            
            # 提取回答和文档
            answer = result.get("answer", "无法生成回答")
            documents = result.get("documents", [])
            
            # 添加助手消息
            assistant_message = self.add_message(
                role="assistant",
                content=answer,
                metadata={"query": query}
            )
            
            # 为回答添加引用
            for doc in documents:
                self.add_citation(
                    message_id=assistant_message.message_id,
                    document_id=doc.get("id", "unknown"),
                    text=doc.get("content", "")[:200] + "...",  # 截取部分内容作为引用文本
                    metadata=doc.get("metadata", {})
                )
            
            logger.info(f"会话{self.session_id}处理查询成功: '{query[:50]}...'")
            return assistant_message, documents
            
        except Exception as e:
            logger.error(f"会话{self.session_id}处理查询失败: {str(e)}")
            
            # 添加错误消息
            error_message = self.add_message(
                role="assistant",
                content=f"处理查询时出错: {str(e)}",
                metadata={"error": str(e), "query": query}
            )
            
            return error_message, []
    
    def get_messages(self, limit: int = 0) -> List[Dict[str, Any]]:
        """
        获取消息列表
        
        Args:
            limit (int): 返回消息的数量限制，0表示返回所有消息
            
        Returns:
            list: 消息字典列表
        """
        messages = [msg.to_dict() for msg in self.messages]
        
        # 添加引用信息
        for msg in messages:
            message_id = msg.get("message_id")
            if message_id in self.citations:
                msg["citations"] = [citation.to_dict() for citation in self.citations[message_id]]
        
        # 限制消息数量
        if limit > 0 and limit < len(messages):
            return messages[-limit:]
        
        return messages
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        获取最后一条消息
        
        Returns:
            dict: 消息字典，如果没有消息则返回None
        """
        if not self.messages:
            return None
        
        message = self.messages[-1].to_dict()
        
        # 添加引用信息
        message_id = message.get("message_id")
        if message_id in self.citations:
            message["citations"] = [citation.to_dict() for citation in self.citations[message_id]]
        
        return message
    
    def clear_history(self):
        """清空历史记录"""
        self.messages = []
        self.citations = {}
        logger.info(f"会话{self.session_id}清空了历史记录")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            dict: 会话字典
        """
        return {
            "session_id": self.session_id,
            "max_history_length": self.max_history_length,
            "metadata": self.metadata,
            "messages": [msg.to_dict() for msg in self.messages],
            "citations": {
                msg_id: [citation.to_dict() for citation in citations]
                for msg_id, citations in self.citations.items()
            },
            "created_at": self.created_at,
            "last_active_at": self.last_active_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], rag_pipeline: Optional[RAGPipeline] = None) -> 'ChatSession':
        """
        从字典创建会话
        
        Args:
            data (dict): 会话字典
            rag_pipeline (RAGPipeline, optional): RAG管道
            
        Returns:
            ChatSession: 会话实例
        """
        session = cls(
            session_id=data.get("session_id"),
            rag_pipeline=rag_pipeline,
            max_history_length=data.get("max_history_length", 10),
            metadata=data.get("metadata", {})
        )
        
        # 添加消息
        session.messages = [
            Message.from_dict(msg) for msg in data.get("messages", [])
        ]
        
        # 添加引用
        for msg_id, citations in data.get("citations", {}).items():
            session.citations[msg_id] = [
                Citation.from_dict(citation) for citation in citations
            ]
        
        # 设置时间戳
        session.created_at = data.get("created_at", session.created_at)
        session.last_active_at = data.get("last_active_at", session.last_active_at)
        
        return session


class ChatSessionManager:
    """聊天会话管理器"""
    
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        max_sessions: int = 100,
        default_max_history_length: int = 10
    ):
        """
        初始化聊天会话管理器
        
        Args:
            rag_pipeline (RAGPipeline, optional): RAG管道
            max_sessions (int): 最大会话数量
            default_max_history_length (int): 默认最大历史记录长度
        """
        self.rag_pipeline = rag_pipeline
        self.max_sessions = max_sessions
        self.default_max_history_length = default_max_history_length
        self.sessions = {}  # 会话ID到会话实例的映射
    
    def create_session(
        self,
        session_id: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_history_length: int = None
    ) -> ChatSession:
        """
        创建会话
        
        Args:
            session_id (str, optional): 会话ID
            metadata (dict, optional): 会话元数据
            max_history_length (int, optional): 最大历史记录长度
            
        Returns:
            ChatSession: 创建的会话
        """
        # 检查会话数量是否达到上限
        if len(self.sessions) >= self.max_sessions:
            # 移除最早的会话
            oldest_session_id = min(
                self.sessions,
                key=lambda sid: self.sessions[sid].last_active_at
            )
            del self.sessions[oldest_session_id]
            logger.info(f"会话数量达到上限，移除了最早的会话: {oldest_session_id}")
        
        # 创建会话
        session = ChatSession(
            session_id=session_id,
            rag_pipeline=self.rag_pipeline,
            max_history_length=max_history_length or self.default_max_history_length,
            metadata=metadata
        )
        
        # 添加到会话映射
        self.sessions[session.session_id] = session
        
        logger.info(f"创建了新会话: {session.session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        获取会话
        
        Args:
            session_id (str): 会话ID
            
        Returns:
            ChatSession: 会话实例，如果不存在则返回None
        """
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            session_id (str): 会话ID
            
        Returns:
            bool: 是否成功删除
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除了会话: {session_id}")
            return True
        
        logger.warning(f"尝试删除不存在的会话: {session_id}")
        return False
    
    def get_all_sessions(self) -> Dict[str, ChatSession]:
        """
        获取所有会话
        
        Returns:
            dict: 会话ID到会话实例的映射
        """
        return self.sessions
    
    def save_sessions(self, file_path: str) -> bool:
        """
        保存会话到文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 转换会话为字典
            sessions_data = {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            }
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功保存{len(sessions_data)}个会话到: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存会话失败: {str(e)}")
            return False
    
    def load_sessions(self, file_path: str) -> bool:
        """
        从文件加载会话
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            # 从文件加载
            with open(file_path, 'r', encoding='utf-8') as f:
                sessions_data = json.load(f)
            
            # 创建会话实例
            self.sessions = {
                session_id: ChatSession.from_dict(
                    data=session_data,
                    rag_pipeline=self.rag_pipeline
                )
                for session_id, session_data in sessions_data.items()
            }
            
            logger.info(f"成功加载{len(self.sessions)}个会话从: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载会话失败: {str(e)}")
            return False
