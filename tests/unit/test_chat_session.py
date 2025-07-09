"""
聊天会话管理单元测试
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

from rag.chat_session import Message, Citation, ChatSession, ChatSessionManager
from rag.haystack_pipeline import RAGPipeline


class TestMessage:
    """聊天消息单元测试类"""
    
    def test_initialization(self):
        """测试初始化"""
        # 基本初始化
        msg = Message(role="user", content="Test message")
        assert msg.role == "user"
        assert msg.content == "Test message"
        assert msg.message_id is not None  # 应该自动生成ID
        assert msg.timestamp is not None  # 应该自动生成时间戳
        assert isinstance(msg.metadata, dict)
        assert len(msg.metadata) == 0
        
        # 带自定义参数的初始化
        custom_id = "custom-id-123"
        custom_time = 1647511234.5678
        custom_meta = {"source": "test", "priority": "high"}
        
        msg = Message(
            role="assistant",
            content="Custom message",
            message_id=custom_id,
            timestamp=custom_time,
            metadata=custom_meta
        )
        
        assert msg.role == "assistant"
        assert msg.content == "Custom message"
        assert msg.message_id == custom_id
        assert msg.timestamp == custom_time
        assert msg.metadata == custom_meta
    
    def test_to_dict(self):
        """测试转换为字典"""
        custom_id = "msg-123"
        custom_time = 1647511234.5678
        custom_meta = {"source": "test"}
        
        msg = Message(
            role="user",
            content="Test content",
            message_id=custom_id,
            timestamp=custom_time,
            metadata=custom_meta
        )
        
        result = msg.to_dict()
        
        assert isinstance(result, dict)
        assert result["message_id"] == custom_id
        assert result["role"] == "user"
        assert result["content"] == "Test content"
        assert result["timestamp"] == custom_time
        assert result["metadata"] == custom_meta
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "message_id": "msg-456",
            "role": "assistant",
            "content": "Dict content",
            "timestamp": 1647522345.6789,
            "metadata": {"format": "markdown"}
        }
        
        msg = Message.from_dict(data)
        
        assert msg.message_id == "msg-456"
        assert msg.role == "assistant"
        assert msg.content == "Dict content"
        assert msg.timestamp == 1647522345.6789
        assert msg.metadata == {"format": "markdown"}
        
        # 测试缺少字段的情况
        incomplete_data = {
            "role": "user",
            "content": "Incomplete data"
        }
        
        msg = Message.from_dict(incomplete_data)
        
        assert msg.role == "user"
        assert msg.content == "Incomplete data"
        assert msg.message_id is not None  # 应该自动生成
        assert msg.timestamp is not None  # 应该自动生成
        assert isinstance(msg.metadata, dict)


class TestCitation:
    """引用单元测试类"""
    
    def test_initialization(self):
        """测试初始化"""
        # 基本初始化
        citation = Citation(document_id="doc-123", text="Citation text")
        
        assert citation.document_id == "doc-123"
        assert citation.text == "Citation text"
        assert isinstance(citation.metadata, dict)
        assert len(citation.metadata) == 0
        
        # 带元数据的初始化
        meta = {"page": 5, "confidence": 0.9}
        citation = Citation(
            document_id="doc-456",
            text="With metadata",
            metadata=meta
        )
        
        assert citation.document_id == "doc-456"
        assert citation.text == "With metadata"
        assert citation.metadata == meta
    
    def test_to_dict(self):
        """测试转换为字典"""
        meta = {"page": 10}
        citation = Citation(
            document_id="doc-789",
            text="To dict test",
            metadata=meta
        )
        
        result = citation.to_dict()
        
        assert isinstance(result, dict)
        assert result["document_id"] == "doc-789"
        assert result["text"] == "To dict test"
        assert result["metadata"] == meta
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "document_id": "doc-abc",
            "text": "From dict test",
            "metadata": {"section": "introduction"}
        }
        
        citation = Citation.from_dict(data)
        
        assert citation.document_id == "doc-abc"
        assert citation.text == "From dict test"
        assert citation.metadata == {"section": "introduction"}
        
        # 测试缺少字段的情况
        incomplete_data = {
            "document_id": "doc-def",
            "text": "Incomplete data"
        }
        
        citation = Citation.from_dict(incomplete_data)
        
        assert citation.document_id == "doc-def"
        assert citation.text == "Incomplete data"
        assert isinstance(citation.metadata, dict)
        assert len(citation.metadata) == 0


class TestChatSession:
    """聊天会话单元测试类"""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """模拟RAG管道"""
        pipeline = MagicMock(spec=RAGPipeline)
        pipeline.run.return_value = {
            "answer": "Test answer",
            "documents": [
                {"id": "doc1", "content": "Doc 1 content", "metadata": {"title": "Doc 1"}},
                {"id": "doc2", "content": "Doc 2 content", "metadata": {"title": "Doc 2"}}
            ]
        }
        return pipeline
    
    def test_initialization(self, mock_rag_pipeline):
        """测试初始化"""
        # 基本初始化
        session = ChatSession()
        
        assert session.session_id is not None  # 应该自动生成ID
        assert session.rag_pipeline is None
        assert session.max_history_length == 10
        assert isinstance(session.metadata, dict)
        assert len(session.metadata) == 0
        assert isinstance(session.messages, list)
        assert len(session.messages) == 0
        assert isinstance(session.citations, dict)
        assert len(session.citations) == 0
        assert session.created_at is not None
        assert session.last_active_at == session.created_at
        
        # 带自定义参数的初始化
        custom_id = "session-123"
        custom_meta = {"user_id": "user-456"}
        
        session = ChatSession(
            session_id=custom_id,
            rag_pipeline=mock_rag_pipeline,
            max_history_length=5,
            metadata=custom_meta
        )
        
        assert session.session_id == custom_id
        assert session.rag_pipeline == mock_rag_pipeline
        assert session.max_history_length == 5
        assert session.metadata == custom_meta
    
    def test_add_message(self):
        """测试添加消息"""
        session = ChatSession(max_history_length=3)
        
        # 添加第一条消息
        msg1 = session.add_message(role="user", content="Message 1")
        
        assert isinstance(msg1, Message)
        assert msg1.role == "user"
        assert msg1.content == "Message 1"
        assert len(session.messages) == 1
        assert session.messages[0] == msg1
        
        # 添加第二条消息
        msg2 = session.add_message(
            role="assistant",
            content="Message 2",
            message_id="custom-id",
            metadata={"important": True}
        )
        
        assert isinstance(msg2, Message)
        assert msg2.role == "assistant"
        assert msg2.content == "Message 2"
        assert msg2.message_id == "custom-id"
        assert msg2.metadata == {"important": True}
        assert len(session.messages) == 2
        assert session.messages[1] == msg2
        
        # 测试历史记录长度限制
        # 添加足够多的消息以触发历史记录裁剪
        for i in range(10):  # 添加10条消息，总共12条
            session.add_message(role="user", content=f"Extra message {i}")
        
        # 由于max_history_length=3，应该只保留最近的6条消息（3对用户-助手消息）
        assert len(session.messages) <= 6
    
    def test_add_citation(self):
        """测试添加引用"""
        session = ChatSession()
        
        # 添加消息
        msg = session.add_message(role="assistant", content="Message with citation")
        
        # 添加引用
        citation = session.add_citation(
            message_id=msg.message_id,
            document_id="doc-123",
            text="Citation text",
            metadata={"page": 5}
        )
        
        assert isinstance(citation, Citation)
        assert citation.document_id == "doc-123"
        assert citation.text == "Citation text"
        assert citation.metadata == {"page": 5}
        
        # 验证引用已添加到会话
        assert msg.message_id in session.citations
        assert len(session.citations[msg.message_id]) == 1
        assert session.citations[msg.message_id][0] == citation
        
        # 添加第二个引用
        citation2 = session.add_citation(
            message_id=msg.message_id,
            document_id="doc-456",
            text="Another citation"
        )
        
        assert len(session.citations[msg.message_id]) == 2
        assert session.citations[msg.message_id][1] == citation2
    
    def test_get_message_citations(self):
        """测试获取消息引用"""
        session = ChatSession()
        
        # 添加消息
        msg = session.add_message(role="assistant", content="Message with citations")
        
        # 添加引用
        citation1 = session.add_citation(
            message_id=msg.message_id,
            document_id="doc-123",
            text="Citation 1"
        )
        citation2 = session.add_citation(
            message_id=msg.message_id,
            document_id="doc-456",
            text="Citation 2"
        )
        
        # 获取引用
        citations = session.get_message_citations(msg.message_id)
        
        assert isinstance(citations, list)
        assert len(citations) == 2
        assert citations[0] == citation1
        assert citations[1] == citation2
        
        # 测试不存在的消息ID
        citations = session.get_message_citations("non-existent-id")
        assert isinstance(citations, list)
        assert len(citations) == 0
    
    def test_get_messages(self):
        """测试获取消息"""
        session = ChatSession()
        
        # 添加消息
        msg1 = session.add_message(role="user", content="User message")
        msg2 = session.add_message(role="assistant", content="Assistant message")
        
        # 获取所有消息
        messages = session.get_messages()
        
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0] == msg1
        assert messages[1] == msg2
        
        # 测试限制数量
        messages = session.get_messages(limit=1)
        
        assert len(messages) == 1
        assert messages[0] == msg2  # 应该返回最新的消息
    
    def test_to_dict(self):
        """测试转换为字典"""
        session = ChatSession(
            session_id="test-session",
            max_history_length=5,
            metadata={"user": "test-user"}
        )
        
        # 添加消息和引用
        msg = session.add_message(role="assistant", content="Test message")
        session.add_citation(message_id=msg.message_id, document_id="doc-123", text="Test citation")
        
        # 转换为字典
        result = session.to_dict()
        
        assert isinstance(result, dict)
        assert result["session_id"] == "test-session"
        assert result["max_history_length"] == 5
        assert result["metadata"] == {"user": "test-user"}
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) == 1
        assert "citations" in result
        assert isinstance(result["citations"], dict)
        assert len(result["citations"]) == 1
        assert "created_at" in result
        assert "last_active_at" in result
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "session_id": "session-from-dict",
            "max_history_length": 7,
            "metadata": {"source": "test"},
            "messages": [
                {
                    "message_id": "msg-1",
                    "role": "user",
                    "content": "User question",
                    "timestamp": 1647511234.5678,
                    "metadata": {}
                },
                {
                    "message_id": "msg-2",
                    "role": "assistant",
                    "content": "Assistant answer",
                    "timestamp": 1647511245.6789,
                    "metadata": {"format": "text"}
                }
            ],
            "citations": {
                "msg-2": [
                    {
                        "document_id": "doc-1",
                        "text": "Citation for answer",
                        "metadata": {"page": 1}
                    }
                ]
            },
            "created_at": 1647511234.5678,
            "last_active_at": 1647511245.6789
        }
        
        session = ChatSession.from_dict(data)
        
        assert session.session_id == "session-from-dict"
        assert session.max_history_length == 7
        assert session.metadata == {"source": "test"}
        assert len(session.messages) == 2
        assert session.messages[0].message_id == "msg-1"
        assert session.messages[1].message_id == "msg-2"
        assert len(session.citations) == 1
        assert "msg-2" in session.citations
        assert len(session.citations["msg-2"]) == 1
        assert session.citations["msg-2"][0].document_id == "doc-1"
        assert session.created_at == 1647511234.5678
        assert session.last_active_at == 1647511245.6789
    
    def test_generate_response(self, mock_rag_pipeline):
        """测试生成回复"""
        session = ChatSession(rag_pipeline=mock_rag_pipeline)
        
        # 添加用户消息
        user_msg = session.add_message(role="user", content="Test question")
        
        # 生成回复
        response = session.generate_response()
        
        # 验证响应
        assert isinstance(response, Message)
        assert response.role == "assistant"
        assert response.content == "Test answer"
        
        # 验证RAG管道被调用
        mock_rag_pipeline.run.assert_called_once()
        
        # 验证引用被添加
        assert response.message_id in session.citations
        assert len(session.citations[response.message_id]) == 2
        assert session.citations[response.message_id][0].document_id == "doc1"
        assert session.citations[response.message_id][1].document_id == "doc2"


class TestChatSessionManager:
    """聊天会话管理器单元测试类"""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """模拟RAG管道"""
        pipeline = MagicMock(spec=RAGPipeline)
        return pipeline
    
    def test_initialization(self, mock_rag_pipeline):
        """测试初始化"""
        manager = ChatSessionManager(
            rag_pipeline=mock_rag_pipeline,
            max_sessions=50,
            default_max_history_length=15
        )
        
        assert manager.rag_pipeline == mock_rag_pipeline
        assert manager.max_sessions == 50
        assert manager.default_max_history_length == 15
        assert isinstance(manager.sessions, dict)
        assert len(manager.sessions) == 0
    
    def test_create_session(self, mock_rag_pipeline):
        """测试创建会话"""
        manager = ChatSessionManager(rag_pipeline=mock_rag_pipeline)
        
        # 创建会话
        session = manager.create_session(
            session_id="test-session",
            metadata={"user": "test-user"},
            max_history_length=20
        )
        
        # 验证会话
        assert isinstance(session, ChatSession)
        assert session.session_id == "test-session"
        assert session.rag_pipeline == mock_rag_pipeline
        assert session.max_history_length == 20
        assert session.metadata == {"user": "test-user"}
        
        # 验证会话已添加到管理器
        assert "test-session" in manager.sessions
        assert manager.sessions["test-session"] == session
        
        # 测试创建不指定ID的会话
        session2 = manager.create_session()
        
        assert isinstance(session2, ChatSession)
        assert session2.session_id is not None
        assert session2.session_id in manager.sessions
        
        # 测试会话数量限制
        manager.max_sessions = 2  # 已创建2个会话，再创建应该删除最旧的
        manager.create_session(session_id="session3")
        
        assert len(manager.sessions) <= 2
        assert "session3" in manager.sessions
    
    def test_get_session(self, mock_rag_pipeline):
        """测试获取会话"""
        manager = ChatSessionManager(rag_pipeline=mock_rag_pipeline)
        
        # 创建会话
        session = manager.create_session(session_id="get-test")
        
        # 获取会话
        retrieved = manager.get_session("get-test")
        
        assert retrieved == session
        
        # 测试获取不存在的会话
        non_existent = manager.get_session("non-existent")
        assert non_existent is None
    
    def test_delete_session(self, mock_rag_pipeline):
        """测试删除会话"""
        manager = ChatSessionManager(rag_pipeline=mock_rag_pipeline)
        
        # 创建会话
        manager.create_session(session_id="delete-test")
        
        # 验证会话存在
        assert "delete-test" in manager.sessions
        
        # 删除会话
        result = manager.delete_session("delete-test")
        
        # 验证删除成功
        assert result is True
        assert "delete-test" not in manager.sessions
        
        # 测试删除不存在的会话
        result = manager.delete_session("non-existent")
        assert result is False
    
    def test_get_all_sessions(self, mock_rag_pipeline):
        """测试获取所有会话"""
        manager = ChatSessionManager(rag_pipeline=mock_rag_pipeline)
        
        # 创建会话
        session1 = manager.create_session(session_id="session1")
        session2 = manager.create_session(session_id="session2")
        
        # 获取所有会话
        sessions = manager.get_all_sessions()
        
        assert isinstance(sessions, dict)
        assert len(sessions) == 2
        assert "session1" in sessions
        assert "session2" in sessions
        assert sessions["session1"] == session1
        assert sessions["session2"] == session2
    
    def test_save_and_load_sessions(self, mock_rag_pipeline):
        """测试保存和加载会话"""
        manager = ChatSessionManager(rag_pipeline=mock_rag_pipeline)
        
        # 创建会话
        session = manager.create_session(session_id="save-test")
        session.add_message(role="user", content="Test message")
        
        # 保存会话到临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        
        result = manager.save_sessions(temp_file)
        assert result is True
        
        # 创建新的管理器
        new_manager = ChatSessionManager(rag_pipeline=mock_rag_pipeline)
        
        # 加载会话
        result = new_manager.load_sessions(temp_file)
        
        # 验证加载成功
        assert result is True
        assert "save-test" in new_manager.sessions
        loaded_session = new_manager.get_session("save-test")
        assert loaded_session.session_id == "save-test"
        assert len(loaded_session.messages) == 1
        assert loaded_session.messages[0].content == "Test message"
        
        # 清理
        if os.path.exists(temp_file):
            os.unlink(temp_file)
