"""
RAG系统集成测试
测试整个检索增强生成系统的端到端功能
"""

import os
import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock
import shutil

from haystack.dataclasses import Document as HaystackDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore

from core.config_manager import ConfigManager
from rag.haystack_pipeline import RAGPipelineFactory
from rag.prompt_builder import PromptBuilderFactory
from rag.chat_session import ChatSessionManager
from connectors.haystack_llm_connector import HaystackLLMFactory
from processors.document_processor import DocumentProcessor
from models.document import Document


class TestRAGIntegration:
    """RAG系统集成测试"""
    
    @pytest.fixture
    def config_file(self):
        """创建临时配置文件"""
        config_content = {
            "rag": {
                "retriever_top_k": 3,
                "prompt_builder": {
                    "template_type": "academic",
                    "max_context_length": 3000,
                    "include_citation": True
                }
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 500
            },
            "document_store": {
                "type": "in_memory",
                "similarity": "cosine"
            },
            "document_processor": {
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }
        
        # 创建临时配置文件
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "config.json")
        
        with open(config_path, "w") as f:
            json.dump(config_content, f)
        
        yield config_path
        
        # 测试后清理
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_documents(self):
        """生成样本文档"""
        return [
            Document("physics.pdf").with_content(
                "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics. "
                "His work is also known for its influence on the philosophy of science."
            ).with_metadata({
                "title": "Einstein and Theory of Relativity",
                "author": "Science Foundation",
                "year": 2020
            }),
            Document("ml.pdf").with_content(
                "Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed. "
                "It is a subset of artificial intelligence focused on developing systems that can learn from data."
            ).with_metadata({
                "title": "Introduction to Machine Learning",
                "author": "AI Research Group",
                "year": 2021
            }),
            Document("quantum.pdf").with_content(
                "Quantum computing is a type of computation that harnesses the collective properties of quantum states. "
                "It uses quantum bits or qubits which can be in multiple states simultaneously."
            ).with_metadata({
                "title": "Quantum Computing Basics",
                "author": "Quantum Physics Institute",
                "year": 2022
            })
        ]
    
    @pytest.fixture
    def mock_llm_api(self):
        """模拟LLM API调用"""
        with patch('connectors.api_connector.APIConnector.make_request') as mock_make_request:
            def side_effect_func(method, endpoint, headers, json_data, **kwargs):
                # 模拟不同查询的LLM响应
                query = json_data.get("messages", [{}])[-1].get("content", "")
                
                if "einstein" in query.lower() or "relativity" in query.lower():
                    response = {
                        "choices": [{
                            "message": {
                                "content": "Einstein developed the theory of relativity in the early 20th century. "
                                           "This theory revolutionized our understanding of space, time, and gravity."
                            }
                        }]
                    }
                elif "machine learning" in query.lower():
                    response = {
                        "choices": [{
                            "message": {
                                "content": "Machine Learning is a field that enables computers to learn from data without explicit programming. "
                                           "It's widely used in various applications including image recognition, natural language processing, and recommendation systems."
                            }
                        }]
                    }
                elif "quantum" in query.lower():
                    response = {
                        "choices": [{
                            "message": {
                                "content": "Quantum computing uses quantum bits (qubits) that can represent multiple states simultaneously. "
                                           "This property allows quantum computers to solve certain problems much faster than classical computers."
                            }
                        }]
                    }
                else:
                    response = {
                        "choices": [{
                            "message": {
                                "content": "I don't have specific information about that in my knowledge base."
                            }
                        }]
                    }
                
                return response
                
            mock_make_request.side_effect = side_effect_func
            yield mock_make_request
    
    @pytest.fixture
    def mock_embeddings_api(self):
        """模拟嵌入向量API调用"""
        with patch('connectors.embedding_connector.OpenAIEmbeddingConnector.get_embeddings') as mock_embeddings:
            # 为不同文本返回模拟的嵌入向量
            def get_mock_embeddings(texts):
                # 简化版，为每个文本返回固定长度的随机值数组
                return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]
            
            mock_embeddings.side_effect = get_mock_embeddings
            yield mock_embeddings
    
    @pytest.fixture
    def document_processor(self):
        """创建文档处理器"""
        return DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    @pytest.fixture
    def rag_system(self, config_file, mock_llm_api, mock_embeddings_api):
        """设置完整的RAG系统"""
        # 加载配置
        config_manager = ConfigManager(config_file)
        
        # 创建文档存储
        document_store = InMemoryDocumentStore(similarity="cosine")
        
        # 创建LLM连接器
        with patch('connectors.haystack_llm_connector.OpenAIConnector'):
            llm_connector = HaystackLLMConnectorFactory.create_connector(
                provider="openai",
                model="gpt-3.5-turbo",
                config=config_manager.get_config()["llm"]
            )
        
        # 创建提示构建器
        prompt_builder = PromptBuilderFactory.create_builder(
            config=config_manager.get_config()["rag"]["prompt_builder"]
        )
        
        # 创建RAG管道
        rag_pipeline = RAGPipelineFactory.create_pipeline(
            llm_connector=llm_connector,
            document_store=document_store,
            prompt_builder=prompt_builder,
            config=config_manager.get_config()["rag"]
        )
        
        # 创建会话管理器
        session_manager = ChatSessionManager(
            rag_pipeline=rag_pipeline,
            default_max_history_length=10
        )
        
        return {
            "config_manager": config_manager,
            "document_store": document_store,
            "llm_connector": llm_connector,
            "prompt_builder": prompt_builder,
            "rag_pipeline": rag_pipeline,
            "session_manager": session_manager
        }
    
    def test_document_processing_to_haystack(self, sample_documents, document_processor, rag_system):
        """测试从原始文档到Haystack文档的处理流程"""
        document_store = rag_system["document_store"]
        
        # 处理样本文档并添加到文档存储
        for doc in sample_documents:
            # 处理文档
            processed = document_processor.process(doc)
            assert processed.is_successful()
            
            # 从处理结果获取Haystack文档
            haystack_docs = processed.get_result("haystack_documents")
            assert haystack_docs is not None
            assert len(haystack_docs) > 0
            
            # 验证处理后的文档结构
            for h_doc in haystack_docs:
                assert isinstance(h_doc, HaystackDocument)
                assert h_doc.content is not None and len(h_doc.content) > 0
                assert h_doc.id is not None
                assert h_doc.metadata is not None
                
                # 确保元数据被正确保留
                for key, value in doc.metadata.items():
                    assert key in h_doc.metadata
                    assert h_doc.metadata[key] == value
            
            # 将文档添加到文档存储
            document_store.write_documents(haystack_docs)
        
        # 验证文档已添加到存储
        all_docs = document_store.filter_documents({})
        assert len(all_docs) > 0
        
        # 应该有至少与原始文档数量相同的文档（可能更多，如果文档被分块）
        assert len(all_docs) >= len(sample_documents)
    
    def test_rag_query_execution(self, sample_documents, document_processor, rag_system, mock_llm_api, mock_embeddings_api):
        """测试RAG查询执行流程"""
        document_store = rag_system["document_store"]
        rag_pipeline = rag_system["rag_pipeline"]
        
        # 处理样本文档并添加到文档存储
        for doc in sample_documents:
            processed = document_processor.process(doc)
            haystack_docs = processed.get_result("haystack_documents")
            document_store.write_documents(haystack_docs)
        
        # 执行查询
        with patch('rag.haystack_pipeline.Pipeline.run') as mock_run:
            # 模拟Pipeline.run的返回
            retrieved_docs = [HaystackDocument(content=doc.content, id=doc.id, metadata=doc.metadata) 
                             for doc in document_store.filter_documents({})]
            
            mock_run.return_value = {
                "llm": {"replies": [MagicMock(content="Einstein developed the theory of relativity.")]},
                "retriever": {"documents": retrieved_docs}
            }
            
            # 执行查询
            result = rag_pipeline.run("Tell me about Einstein and relativity")
            
            # 验证结果
            assert "answer" in result
            assert "documents" in result
            assert result["answer"] == "Einstein developed the theory of relativity."
            assert len(result["documents"]) > 0
    
    def test_chat_session_with_rag(self, sample_documents, document_processor, rag_system, mock_llm_api, mock_embeddings_api):
        """测试使用RAG的聊天会话"""
        document_store = rag_system["document_store"]
        session_manager = rag_system["session_manager"]
        
        # 处理样本文档并添加到文档存储
        for doc in sample_documents:
            processed = document_processor.process(doc)
            haystack_docs = processed.get_result("haystack_documents")
            document_store.write_documents(haystack_docs)
        
        # 创建会话
        with patch('rag.haystack_pipeline.Pipeline.run') as mock_run:
            # 模拟Pipeline.run的返回
            retrieved_docs = [HaystackDocument(
                content="Albert Einstein developed the theory of relativity.", 
                id="doc1",
                metadata={"title": "Einstein and Theory of Relativity"}
            )]
            
            mock_run.return_value = {
                "llm": {"replies": [MagicMock(content="Einstein developed the theory of relativity.")]},
                "retriever": {"documents": retrieved_docs}
            }
            
            # 创建会话
            session = session_manager.create_session(session_id="integration-test")
            
            # 添加用户消息
            user_msg = session.add_message(role="user", content="Tell me about Einstein")
            
            # 生成回复
            response = session.generate_response()
            
            # 验证回复
            assert response.role == "assistant"
            assert "Einstein" in response.content
            assert "relativity" in response.content.lower()
            
            # 验证引用
            assert response.message_id in session.citations
            assert len(session.citations[response.message_id]) > 0
    
    def test_multi_turn_conversation(self, sample_documents, document_processor, rag_system, mock_llm_api, mock_embeddings_api):
        """测试多轮对话的上下文保持"""
        document_store = rag_system["document_store"]
        session_manager = rag_system["session_manager"]
        
        # 处理样本文档并添加到文档存储
        for doc in sample_documents:
            processed = document_processor.process(doc)
            haystack_docs = processed.get_result("haystack_documents")
            document_store.write_documents(haystack_docs)
        
        with patch('rag.haystack_pipeline.Pipeline.run') as mock_run:
            # 第一轮的模拟返回
            mock_run.return_value = {
                "llm": {"replies": [MagicMock(content="Einstein developed the theory of relativity.")]},
                "retriever": {"documents": [HaystackDocument(
                    content="Albert Einstein developed the theory of relativity.", 
                    id="doc1",
                    metadata={"title": "Einstein and Theory of Relativity"}
                )]}
            }
            
            # 创建会话
            session = session_manager.create_session(session_id="multi-turn-test")
            
            # 第一轮对话
            session.add_message(role="user", content="Tell me about Einstein")
            response1 = session.generate_response()
            
            assert "Einstein" in response1.content
            
            # 为第二轮设置不同的模拟返回
            mock_run.return_value = {
                "llm": {"replies": [MagicMock(content="The theory of relativity revolutionized physics.")]},
                "retriever": {"documents": [HaystackDocument(
                    content="The theory of relativity revolutionized our understanding of space and time.", 
                    id="doc2",
                    metadata={"title": "Modern Physics"}
                )]}
            }
            
            # 第二轮对话（追问）
            session.add_message(role="user", content="What impact did his theory have?")
            response2 = session.generate_response()
            
            # 验证第二轮回复
            assert "revolutionized" in response2.content.lower()
            
            # 验证模拟Pipeline.run被调用两次，第二次应该包含聊天历史
            assert mock_run.call_count == 2
    
    def test_error_handling(self, rag_system):
        """测试错误处理情况"""
        session_manager = rag_system["session_manager"]
        rag_pipeline = rag_system["rag_pipeline"]
        
        # 设置RAG管道抛出异常
        rag_pipeline.run.side_effect = Exception("测试错误")
        
        # 创建会话
        session = session_manager.create_session(session_id="error-test")
        
        # 添加用户消息
        session.add_message(role="user", content="This will cause an error")
        
        # 生成回复（应该处理错误）
        response = session.generate_response()
        
        # 验证错误处理
        assert response.role == "assistant"
        assert "查询处理失败" in response.content or "抱歉" in response.content.lower()
        
        # 确保系统在错误后仍能正常运行
        rag_pipeline.run.side_effect = None  # 清除错误
        rag_pipeline.run.return_value = {
            "answer": "正常回复",
            "documents": [],
            "query": "Next query"
        }
        
        # 添加新消息
        session.add_message(role="user", content="This should work now")
        response2 = session.generate_response()
        
        # 验证系统恢复
        assert "正常回复" in response2.content
    
    def test_session_persistence_and_recovery(self, rag_system):
        """测试会话持久化和恢复"""
        session_manager = rag_system["session_manager"]
        rag_pipeline = rag_system["rag_pipeline"]
        
        # 设置RAG管道返回
        rag_pipeline.run.return_value = {
            "answer": "测试回复",
            "documents": [{"id": "test-doc", "content": "测试内容", "metadata": {}}],
            "query": "测试查询"
        }
        
        # 创建会话并添加内容
        session = session_manager.create_session(session_id="persistence-test")
        session.add_message(role="user", content="Test message")
        response = session.generate_response()
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        
        try:
            # 保存会话
            session_manager.save_sessions(temp_file)
            
            # 创建新的会话管理器
            new_manager = ChatSessionManager(
                rag_pipeline=rag_pipeline,
                default_max_history_length=10
            )
            
            # 加载会话
            new_manager.load_sessions(temp_file)
            
            # 验证会话加载
            loaded_session = new_manager.get_session("persistence-test")
            assert loaded_session is not None
            assert loaded_session.session_id == "persistence-test"
            
            # 验证消息和引用被正确加载
            messages = loaded_session.get_messages()
            assert len(messages) == 2
            assert messages[0].content == "Test message"
            assert messages[1].content == response.content
            
            # 验证引用被正确加载
            assert messages[1].message_id in loaded_session.citations
            
            # 验证加载后的会话可以继续使用
            loaded_session.add_message(role="user", content="Follow-up question")
            follow_up_response = loaded_session.generate_response()
            
            assert follow_up_response.role == "assistant"
            assert follow_up_response.content == "测试回复"
            
        finally:
            # 清理
            if os.path.exists(temp_file):
                os.unlink(temp_file)
