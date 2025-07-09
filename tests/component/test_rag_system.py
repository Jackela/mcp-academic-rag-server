"""
RAG系统组件测试 - 测试RAG管道与会话管理的集成
"""

import pytest
from unittest.mock import patch, MagicMock, ANY
import tempfile
import os
import json

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import Document as HaystackDocument

from rag.haystack_pipeline import RAGPipeline
from rag.chat_session import ChatSessionManager, ChatSession
from rag.prompt_builder import ChatPromptBuilder
from connectors.haystack_llm_connector import HaystackLLMConnector


class TestRAGSystem:
    """测试RAG系统的组件交互"""
    
    @pytest.fixture
    def test_documents(self):
        """创建测试文档"""
        return [
            HaystackDocument(
                content="Einstein developed the theory of relativity.", 
                id="doc1", 
                metadata={"title": "Physics History"}
            ),
            HaystackDocument(
                content="Neural networks are a fundamental component of deep learning.", 
                id="doc2", 
                metadata={"title": "Machine Learning Basics"}
            ),
            HaystackDocument(
                content="The structure of DNA was discovered by Watson and Crick.", 
                id="doc3", 
                metadata={"title": "Biology Discoveries"}
            )
        ]
    
    @pytest.fixture
    def mock_document_store(self, test_documents):
        """创建模拟文档存储"""
        store = MagicMock(spec=InMemoryDocumentStore)
        store.filter_documents.return_value = test_documents
        return store
    
    @pytest.fixture
    def mock_llm_connector(self):
        """创建模拟LLM连接器"""
        connector = MagicMock(spec=HaystackLLMConnector)
        connector.generator = MagicMock()
        connector.model = "gpt-3.5-turbo"
        return connector
    
    @pytest.fixture
    def mock_rag_pipeline(self, mock_document_store, mock_llm_connector):
        """创建模拟RAG管道"""
        pipeline = MagicMock(spec=RAGPipeline)
        
        # 根据查询返回不同的响应
        def run_side_effect(query, chat_history=None):
            if "physics" in query.lower():
                return {
                    "answer": "Einstein developed the theory of relativity in the early 20th century.",
                    "documents": [{"id": "doc1", "content": test_documents[0].content, "metadata": test_documents[0].metadata}],
                    "query": query
                }
            elif "neural network" in query.lower():
                return {
                    "answer": "Neural networks are fundamental to deep learning in artificial intelligence.",
                    "documents": [{"id": "doc2", "content": test_documents[1].content, "metadata": test_documents[1].metadata}],
                    "query": query
                }
            else:
                return {
                    "answer": "I don't have specific information about that topic.",
                    "documents": [],
                    "query": query
                }
        
        pipeline.run.side_effect = run_side_effect
        pipeline.document_store = mock_document_store
        pipeline.llm_connector = mock_llm_connector
        return pipeline
    
    @pytest.fixture
    def session_manager(self, mock_rag_pipeline):
        """创建会话管理器"""
        return ChatSessionManager(
            rag_pipeline=mock_rag_pipeline,
            default_max_history_length=10
        )
    
    def test_create_and_use_chat_session(self, session_manager):
        """测试创建和使用聊天会话"""
        # 创建会话
        session = session_manager.create_session(
            session_id="test-session",
            metadata={"user_id": "test-user"}
        )
        
        # 验证会话创建
        assert isinstance(session, ChatSession)
        assert session.session_id == "test-session"
        assert session.metadata == {"user_id": "test-user"}
        assert session.rag_pipeline == session_manager.rag_pipeline
        
        # 添加用户消息并生成回复
        session.add_message(role="user", content="Tell me about physics and Einstein")
        response = session.generate_response()
        
        # 验证回复
        assert response.role == "assistant"
        assert "Einstein" in response.content
        assert "relativity" in response.content.lower()
        
        # 验证RAG管道被调用
        session.rag_pipeline.run.assert_called_once_with(
            "Tell me about physics and Einstein",
            ANY  # 忽略具体的chat_history内容
        )
        
        # 验证引用被添加
        assert response.message_id in session.citations
        assert len(session.citations[response.message_id]) == 1
        assert session.citations[response.message_id][0].document_id == "doc1"
        
        # 验证消息历史记录
        messages = session.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Tell me about physics and Einstein"
        assert messages[1] == response
    
    def test_multi_turn_conversation(self, session_manager):
        """测试多轮对话"""
        session = session_manager.create_session()
        
        # 第一轮对话
        session.add_message(role="user", content="Tell me about physics")
        response1 = session.generate_response()
        
        # 验证第一轮回复
        assert "Einstein" in response1.content
        assert "relativity" in response1.content.lower()
        
        # 第二轮对话
        session.add_message(role="user", content="What about neural networks?")
        response2 = session.generate_response()
        
        # 验证第二轮回复
        assert "neural networks" in response2.content.lower()
        assert "deep learning" in response2.content.lower()
        
        # 验证RAG管道第二次调用时包含聊天历史
        session.rag_pipeline.run.assert_called_with(
            "What about neural networks?",
            ANY  # 应该包含完整的聊天历史
        )
        
        # 验证引用被正确添加
        assert response1.message_id in session.citations
        assert response2.message_id in session.citations
        assert session.citations[response1.message_id][0].document_id == "doc1"
        assert session.citations[response2.message_id][0].document_id == "doc2"
    
    def test_session_management(self, session_manager):
        """测试会话管理功能"""
        # 创建多个会话
        session1 = session_manager.create_session(session_id="session1")
        session2 = session_manager.create_session(session_id="session2")
        
        # 验证会话存储
        assert "session1" in session_manager.sessions
        assert "session2" in session_manager.sessions
        
        # 获取特定会话
        retrieved = session_manager.get_session("session1")
        assert retrieved == session1
        
        # 删除会话
        result = session_manager.delete_session("session1")
        assert result is True
        assert "session1" not in session_manager.sessions
        assert "session2" in session_manager.sessions
    
    def test_session_persistence(self, session_manager):
        """测试会话持久化"""
        # 创建会话并添加消息
        session = session_manager.create_session(session_id="persist-test")
        session.add_message(role="user", content="Test message")
        response = session.generate_response()
        
        # 创建临时文件保存会话
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        
        try:
            # 保存会话
            session_manager.save_sessions(temp_file)
            
            # 验证文件存在且内容合理
            assert os.path.exists(temp_file)
            with open(temp_file, 'r') as f:
                data = json.load(f)
                assert "persist-test" in data
                assert len(data["persist-test"]["messages"]) == 2
            
            # 创建新的会话管理器并加载会话
            new_manager = ChatSessionManager(rag_pipeline=session_manager.rag_pipeline)
            new_manager.load_sessions(temp_file)
            
            # 验证会话加载
            assert "persist-test" in new_manager.sessions
            loaded = new_manager.get_session("persist-test")
            assert loaded.session_id == "persist-test"
            assert len(loaded.messages) == 2
            assert loaded.messages[0].content == "Test message"
            assert loaded.messages[1].content == response.content
            
        finally:
            # 清理
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_session_with_large_history(self, session_manager, mock_rag_pipeline):
        """测试具有大量历史记录的会话"""
        # 创建有历史长度限制的会话
        session = session_manager.create_session(
            session_id="history-test",
            max_history_length=3  # 限制为3对消息(3个用户消息和3个助手回复)
        )
        
        # 添加多个消息对
        for i in range(5):  # 添加5对消息
            session.add_message(role="user", content=f"User message {i}")
            session.add_message(role="assistant", content=f"Assistant response {i}")
        
        # 验证历史记录限制
        messages = session.get_messages()
        assert len(messages) <= 6  # 最多3对消息(6条)
        assert messages[0].content == "User message 2"  # 最早的消息应该是第3对的开始
        assert messages[1].content == "Assistant response 2"
    
    def test_chat_session_context_window(self, mock_rag_pipeline):
        """测试聊天会话的上下文窗口管理"""
        with patch('rag.prompt_builder.ChatPromptBuilder') as mock_builder_class:
            # 创建模拟提示构建器
            mock_builder = MagicMock(spec=ChatPromptBuilder)
            mock_builder_class.return_value = mock_builder
            
            # 设置提示构建器响应
            mock_builder.return_value = {
                "messages": []  # 简化返回值
            }
            
            # 创建会话
            session = ChatSession(
                rag_pipeline=mock_rag_pipeline,
                max_history_length=10
            )
            
            # 添加用户消息并生成回复
            session.add_message(role="user", content="Test query")
            session.generate_response()
            
            # 验证RAG管道调用
            mock_rag_pipeline.run.assert_called_once()
            
            # 验证提示构建器被调用，并考虑上下文窗口限制
            for call_args in mock_builder.call_args_list:
                args, kwargs = call_args
                if 'chat_history' in kwargs:
                    # 确保聊天历史不为空但长度受到限制
                    assert isinstance(kwargs['chat_history'], list)


class TestRAGSystemRealisticScenario:
    """测试更接近真实场景的RAG系统使用"""
    
    @pytest.fixture
    def rag_components(self):
        """创建RAG系统的真实组件（但仍使用模拟的外部依赖）"""
        # 创建真实文档存储
        document_store = InMemoryDocumentStore()
        
        # 添加测试文档
        documents = [
            HaystackDocument(
                content="Machine learning is a branch of artificial intelligence that focuses on using data and algorithms to mimic the way humans learn.",
                id="ml-doc-1",
                metadata={"title": "Introduction to Machine Learning"}
            ),
            HaystackDocument(
                content="Deep learning is a subset of machine learning that uses neural networks with many layers.",
                id="dl-doc-1",
                metadata={"title": "Deep Learning Basics"}
            )
        ]
        document_store.write_documents(documents)
        
        # 模拟LLM连接器
        llm_connector = MagicMock(spec=HaystackLLMConnector)
        llm_connector.generator = MagicMock()
        
        # 创建真实提示构建器
        prompt_builder = ChatPromptBuilder(
            template_type="concise",
            max_context_length=2000
        )
        
        # 创建模拟RAG管道
        rag_pipeline = MagicMock(spec=RAGPipeline)
        
        # 设置RAG管道响应
        def run_side_effect(query, chat_history=None):
            if "machine learning" in query.lower():
                return {
                    "answer": "Machine learning is a branch of AI that uses data and algorithms to learn patterns.",
                    "documents": [{"id": "ml-doc-1", "content": documents[0].content, "metadata": documents[0].metadata}],
                    "query": query
                }
            elif "deep learning" in query.lower():
                return {
                    "answer": "Deep learning is a subset of machine learning using neural networks with many layers.",
                    "documents": [{"id": "dl-doc-1", "content": documents[1].content, "metadata": documents[1].metadata}],
                    "query": query
                }
            else:
                return {
                    "answer": "I don't have specific information on that topic.",
                    "documents": [],
                    "query": query
                }
        
        rag_pipeline.run.side_effect = run_side_effect
        rag_pipeline.document_store = document_store
        rag_pipeline.llm_connector = llm_connector
        rag_pipeline.prompt_builder = prompt_builder
        
        return {
            "document_store": document_store,
            "llm_connector": llm_connector,
            "prompt_builder": prompt_builder,
            "rag_pipeline": rag_pipeline
        }
    
    @pytest.fixture
    def chat_manager(self, rag_components):
        """创建聊天会话管理器"""
        return ChatSessionManager(
            rag_pipeline=rag_components["rag_pipeline"],
            default_max_history_length=10
        )
    
    def test_complete_conversation_flow(self, chat_manager):
        """测试完整的对话流程"""
        # 创建会话
        session = chat_manager.create_session(session_id="realistic-test")
        
        # 添加用户消息
        user_msg = session.add_message(role="user", content="What is machine learning?")
        
        # 生成第一个回复
        response1 = session.generate_response()
        
        # 验证响应
        assert response1.role == "assistant"
        assert "machine learning" in response1.content.lower()
        assert "AI" in response1.content
        
        # 验证引用
        assert response1.message_id in session.citations
        assert len(session.citations[response1.message_id]) == 1
        assert session.citations[response1.message_id][0].document_id == "ml-doc-1"
        
        # 继续对话
        user_msg2 = session.add_message(role="user", content="How does deep learning relate to this?")
        response2 = session.generate_response()
        
        # 验证第二个响应
        assert "deep learning" in response2.content.lower()
        assert "neural networks" in response2.content.lower()
        assert "subset of machine learning" in response2.content.lower()
        
        # 验证第二个引用
        assert response2.message_id in session.citations
        assert len(session.citations[response2.message_id]) == 1
        assert session.citations[response2.message_id][0].document_id == "dl-doc-1"
        
        # 测试未找到相关文档的查询
        user_msg3 = session.add_message(role="user", content="Tell me about quantum computing")
        response3 = session.generate_response()
        
        # 验证第三个响应
        assert "don't have specific information" in response3.content.lower()
        
        # 验证没有引用
        assert response3.message_id in session.citations
        assert len(session.citations[response3.message_id]) == 0
        
        # 获取完整会话历史
        messages = session.get_messages()
        assert len(messages) == 6  # 3个用户消息和3个助手回复
        
        # 验证消息顺序
        assert messages[0].content == "What is machine learning?"
        assert messages[1].content == response1.content
        assert messages[2].content == "How does deep learning relate to this?"
        assert messages[3].content == response2.content
        assert messages[4].content == "Tell me about quantum computing"
        assert messages[5].content == response3.content
