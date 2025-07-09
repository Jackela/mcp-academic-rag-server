"""
Haystack RAG管道单元测试
"""

import pytest
from unittest.mock import patch, MagicMock
from haystack.dataclasses import Document as HaystackDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore

from rag.haystack_pipeline import RAGPipeline, RAGPipelineFactory
from connectors.haystack_llm_connector import HaystackLLMConnector


class TestRAGPipeline:
    """RAG管道单元测试类"""
    
    @pytest.fixture
    def mock_llm_connector(self):
        """模拟LLM连接器"""
        connector = MagicMock(spec=HaystackLLMConnector)
        connector.generator = MagicMock()
        connector.model = "gpt-3.5-turbo"
        return connector
    
    @pytest.fixture
    def mock_document_store(self):
        """模拟文档存储"""
        store = MagicMock(spec=InMemoryDocumentStore)
        return store
    
    @pytest.fixture
    def mock_retriever(self):
        """模拟检索器"""
        retriever = MagicMock()
        return retriever
    
    @pytest.fixture
    def mock_prompt_builder(self):
        """模拟提示构建器"""
        builder = MagicMock()
        return builder
    
    @pytest.fixture
    def rag_pipeline(self, mock_llm_connector, mock_document_store, mock_retriever, mock_prompt_builder):
        """创建RAG管道实例"""
        # 替换InMemoryEmbeddingRetriever
        with patch('rag.haystack_pipeline.InMemoryEmbeddingRetriever', return_value=mock_retriever):
            # 创建管道
            pipeline = RAGPipeline(
                llm_connector=mock_llm_connector,
                document_store=mock_document_store,
                retriever_top_k=3,
                prompt_builder=mock_prompt_builder
            )
            
            # 替换haystack Pipeline
            pipeline.pipeline = MagicMock()
            
            return pipeline
    
    def test_initialization(self, mock_llm_connector, mock_document_store, mock_retriever, mock_prompt_builder):
        """测试初始化过程"""
        # 替换InMemoryEmbeddingRetriever和Pipeline
        with patch('rag.haystack_pipeline.InMemoryEmbeddingRetriever', return_value=mock_retriever), \
             patch('rag.haystack_pipeline.Pipeline') as mock_pipeline_class:
            
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # 创建管道
            pipeline = RAGPipeline(
                llm_connector=mock_llm_connector,
                document_store=mock_document_store,
                retriever_top_k=3,
                prompt_builder=mock_prompt_builder
            )
            
            # 验证检索器创建
            assert pipeline.retriever == mock_retriever
            assert pipeline.retriever_top_k == 3
            assert pipeline.document_store == mock_document_store
            assert pipeline.llm_connector == mock_llm_connector
            assert pipeline.prompt_builder == mock_prompt_builder
            
            # 验证Pipeline创建和组件连接
            mock_pipeline_class.assert_called_once()
            mock_pipeline.add_component.assert_called()
            mock_pipeline.connect.assert_called()
    
    def test_run(self, rag_pipeline, mock_retriever):
        """测试运行RAG管道"""
        # 模拟Pipeline.run的返回值
        mock_documents = [
            HaystackDocument(content="Document 1 content", id="doc1", metadata={"title": "Doc 1"}),
            HaystackDocument(content="Document 2 content", id="doc2", metadata={"title": "Doc 2"})
        ]
        
        rag_pipeline.pipeline.run.return_value = {
            "llm": {"replies": [MagicMock(content="Generated answer")]},
            "retriever": {"documents": mock_documents}
        }
        
        # 运行管道
        query = "测试查询"
        result = rag_pipeline.run(query)
        
        # 验证Pipeline.run被调用
        rag_pipeline.pipeline.run.assert_called_once()
        
        # 验证结果格式
        assert "answer" in result
        assert result["answer"] == "Generated answer"
        assert "documents" in result
        assert len(result["documents"]) == 2
        assert result["documents"][0]["id"] == "doc1"
        assert result["documents"][1]["id"] == "doc2"
        assert "query" in result
        assert result["query"] == query
    
    def test_run_with_error(self, rag_pipeline):
        """测试运行出错的情况"""
        # 模拟Pipeline.run抛出异常
        rag_pipeline.pipeline.run.side_effect = Exception("测试错误")
        
        # 运行管道
        query = "测试查询"
        result = rag_pipeline.run(query)
        
        # 验证错误处理
        assert "answer" in result
        assert "查询处理失败" in result["answer"]
        assert "documents" in result
        assert len(result["documents"]) == 0
        assert "error" in result
        assert "测试错误" in result["error"]
    
    def test_update_retriever(self, rag_pipeline, mock_retriever):
        """测试更新检索器"""
        # 创建模拟新文档存储
        new_document_store = MagicMock(spec=InMemoryDocumentStore)
        
        # 替换InMemoryEmbeddingRetriever
        with patch('rag.haystack_pipeline.InMemoryEmbeddingRetriever', return_value=mock_retriever):
            # 更新检索器
            rag_pipeline.update_retriever(document_store=new_document_store, top_k=5)
            
            # 验证更新
            assert rag_pipeline.document_store == new_document_store
            assert rag_pipeline.retriever_top_k == 5
            assert rag_pipeline.retriever == mock_retriever
    
    def test_update_llm_connector(self, rag_pipeline, mock_llm_connector):
        """测试更新LLM连接器"""
        # 创建模拟新连接器
        new_connector = MagicMock(spec=HaystackLLMConnector)
        new_connector.generator = MagicMock()
        new_connector.model = "gpt-4"
        
        # 更新连接器
        rag_pipeline.update_llm_connector(new_connector)
        
        # 验证更新
        assert rag_pipeline.llm_connector == new_connector


class TestRAGPipelineFactory:
    """RAG管道工厂单元测试类"""
    
    @pytest.fixture
    def mock_llm_connector(self):
        """模拟LLM连接器"""
        connector = MagicMock(spec=HaystackLLMConnector)
        connector.generator = MagicMock()
        connector.model = "gpt-3.5-turbo"
        return connector
    
    def test_create_pipeline(self, mock_llm_connector):
        """测试创建RAG管道"""
        with patch('rag.haystack_pipeline.RAGPipeline') as mock_pipeline_class:
            # 创建模拟管道实例
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # 调用工厂方法
            config = {"retriever_top_k": 7, "extra_param": "value"}
            pipeline = RAGPipelineFactory.create_pipeline(
                llm_connector=mock_llm_connector,
                retriever_top_k=5,
                config=config
            )
            
            # 验证RAGPipeline创建
            mock_pipeline_class.assert_called_once_with(
                llm_connector=mock_llm_connector,
                document_store=None,
                retriever_top_k=7,  # 从config中获取的值
                prompt_builder=None
            )
            
            # 验证返回值
            assert pipeline == mock_pipeline
