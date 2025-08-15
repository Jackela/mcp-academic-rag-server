"""
基础向量存储测试

测试BaseVectorStore抽象基类和通用功能。
"""

import pytest
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, patch
import numpy as np

from haystack import Document as HaystackDocument

from document_stores.implementations.base_vector_store import (
    BaseVectorStore, 
    VectorStoreError, 
    VectorStoreConnectionError,
    VectorStoreOperationError,
    VectorStoreConfigError
)


class MockVectorStore(BaseVectorStore):
    """测试用的模拟向量存储实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.documents = {}
        self.embeddings = {}
        self.next_id = 1
        
    def initialize(self) -> bool:
        self.is_initialized = True
        return True
    
    def add_documents(
        self, 
        documents: List[HaystackDocument],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        if not self.is_initialized:
            return False
            
        for i, doc in enumerate(documents):
            doc_id = doc.id or f"doc_{self.next_id}"
            doc.id = doc_id
            self.documents[doc_id] = doc
            
            if embeddings and i < len(embeddings):
                self.embeddings[doc_id] = embeddings[i]
            elif doc.embedding:
                self.embeddings[doc_id] = doc.embedding
            
            self.next_id += 1
        
        return True
    
    def search(
        self, 
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[HaystackDocument, float]]:
        if not self.is_initialized:
            return []
        
        results = []
        for doc_id, doc in self.documents.items():
            if doc_id in self.embeddings:
                # 简单的相似度计算
                embedding = self.embeddings[doc_id]
                similarity = np.dot(query_embedding, embedding)
                results.append((doc, float(similarity)))
        
        # 按相似度排序并返回top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_document_by_id(self, doc_id: str) -> Optional[HaystackDocument]:
        return self.documents.get(doc_id)
    
    def update_document(
        self, 
        doc_id: str,
        document: HaystackDocument,
        embedding: Optional[List[float]] = None
    ) -> bool:
        if doc_id in self.documents:
            self.documents[doc_id] = document
            if embedding:
                self.embeddings[doc_id] = embedding
            return True
        return False
    
    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.embeddings:
                del self.embeddings[doc_id]
            return True
        return False
    
    def delete_all_documents(self) -> bool:
        self.documents.clear()
        self.embeddings.clear()
        return True
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def save_index(self, path: str) -> bool:
        return True
    
    def load_index(self, path: str) -> bool:
        return True


class TestBaseVectorStore:
    """BaseVectorStore基类测试"""
    
    def test_initialization(self):
        """测试基类初始化"""
        config = {
            "vector_dimension": 384,
            "similarity": "cosine"
        }
        
        store = MockVectorStore(config)
        
        assert store.config == config
        assert store.vector_dim == 384
        assert store.similarity_function == "cosine"
        assert not store.is_initialized
    
    def test_default_config_values(self):
        """测试默认配置值"""
        store = MockVectorStore({})
        
        assert store.vector_dim == 384  # 默认值
        assert store.similarity_function == "dot_product"  # 默认值
    
    def test_validate_embedding_valid(self):
        """测试有效向量嵌入验证"""
        store = MockVectorStore({"vector_dimension": 3})
        
        valid_embedding = [0.1, 0.2, 0.3]
        assert store.validate_embedding(valid_embedding) == True
    
    def test_validate_embedding_wrong_dimension(self):
        """测试错误维度的向量嵌入"""
        store = MockVectorStore({"vector_dimension": 3})
        
        wrong_dim_embedding = [0.1, 0.2]  # 维度不匹配
        assert store.validate_embedding(wrong_dim_embedding) == False
    
    def test_validate_embedding_empty(self):
        """测试空向量嵌入"""
        store = MockVectorStore({"vector_dimension": 3})
        
        assert store.validate_embedding([]) == False
        assert store.validate_embedding(None) == False
    
    def test_validate_embedding_invalid_values(self):
        """测试包含无效值的向量"""
        store = MockVectorStore({"vector_dimension": 3})
        
        # NaN值
        nan_embedding = [0.1, float('nan'), 0.3]
        assert store.validate_embedding(nan_embedding) == False
        
        # 无穷大值
        inf_embedding = [0.1, float('inf'), 0.3]
        assert store.validate_embedding(inf_embedding) == False
    
    def test_get_storage_info(self):
        """测试获取存储信息"""
        store = MockVectorStore({"vector_dimension": 128})
        store.initialize()
        
        info = store.get_storage_info()
        
        assert info["storage_type"] == "MockVectorStore"
        assert info["vector_dimension"] == 128
        assert info["is_initialized"] == True
        assert info["document_count"] == 0
    
    def test_get_supported_similarity_functions(self):
        """测试支持的相似度函数列表"""
        store = MockVectorStore({})
        
        functions = store.get_supported_similarity_functions()
        expected = ["dot_product", "cosine", "euclidean"]
        
        assert set(functions) == set(expected)
    
    def test_close_method(self):
        """测试关闭方法"""
        store = MockVectorStore({})
        store.initialize()
        
        assert store.is_initialized == True
        
        store.close()
        
        assert store.is_initialized == False


class TestVectorStoreOperations:
    """向量存储操作测试"""
    
    @pytest.fixture
    def mock_store(self):
        """创建测试用的向量存储"""
        store = MockVectorStore({"vector_dimension": 3})
        store.initialize()
        return store
    
    @pytest.fixture
    def sample_documents(self):
        """创建示例文档"""
        docs = [
            HaystackDocument(
                content="This is document 1",
                meta={"title": "Doc 1"},
                id="doc1",
                embedding=[0.1, 0.2, 0.3]
            ),
            HaystackDocument(
                content="This is document 2", 
                meta={"title": "Doc 2"},
                id="doc2",
                embedding=[0.4, 0.5, 0.6]
            )
        ]
        return docs
    
    def test_add_documents_success(self, mock_store, sample_documents):
        """测试成功添加文档"""
        result = mock_store.add_documents(sample_documents)
        
        assert result == True
        assert mock_store.get_document_count() == 2
        
        # 验证文档存在
        doc1 = mock_store.get_document_by_id("doc1")
        assert doc1 is not None
        assert doc1.content == "This is document 1"
    
    def test_add_documents_with_external_embeddings(self, mock_store, sample_documents):
        """测试使用外部向量嵌入添加文档"""
        embeddings = [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        
        # 移除文档自带的嵌入
        for doc in sample_documents:
            doc.embedding = None
        
        result = mock_store.add_documents(sample_documents, embeddings)
        
        assert result == True
        assert mock_store.get_document_count() == 2
        
        # 验证使用了外部嵌入
        assert mock_store.embeddings["doc1"] == [0.7, 0.8, 0.9]
    
    def test_add_documents_not_initialized(self, sample_documents):
        """测试未初始化时添加文档"""
        store = MockVectorStore({})  # 未初始化
        
        result = store.add_documents(sample_documents)
        
        assert result == False
    
    def test_search_documents(self, mock_store, sample_documents):
        """测试搜索文档"""
        mock_store.add_documents(sample_documents)
        
        query_embedding = [0.1, 0.2, 0.3]
        results = mock_store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        # 结果应该包含(文档, 相似度得分)元组
        for doc, score in results:
            assert isinstance(doc, HaystackDocument)
            assert isinstance(score, float)
    
    def test_search_empty_store(self, mock_store):
        """测试在空存储中搜索"""
        query_embedding = [0.1, 0.2, 0.3]
        results = mock_store.search(query_embedding)
        
        assert len(results) == 0
    
    def test_get_document_by_id_exists(self, mock_store, sample_documents):
        """测试获取存在的文档"""
        mock_store.add_documents(sample_documents)
        
        doc = mock_store.get_document_by_id("doc1")
        
        assert doc is not None
        assert doc.id == "doc1"
        assert doc.content == "This is document 1"
    
    def test_get_document_by_id_not_exists(self, mock_store):
        """测试获取不存在的文档"""
        doc = mock_store.get_document_by_id("nonexistent")
        
        assert doc is None
    
    def test_update_document_success(self, mock_store, sample_documents):
        """测试成功更新文档"""
        mock_store.add_documents(sample_documents)
        
        updated_doc = HaystackDocument(
            content="Updated document 1",
            meta={"title": "Updated Doc 1"},
            id="doc1"
        )
        
        result = mock_store.update_document("doc1", updated_doc, [0.9, 0.8, 0.7])
        
        assert result == True
        
        # 验证更新
        retrieved_doc = mock_store.get_document_by_id("doc1")
        assert retrieved_doc.content == "Updated document 1"
        assert mock_store.embeddings["doc1"] == [0.9, 0.8, 0.7]
    
    def test_update_document_not_exists(self, mock_store):
        """测试更新不存在的文档"""
        doc = HaystackDocument(content="New doc", id="newdoc")
        
        result = mock_store.update_document("nonexistent", doc)
        
        assert result == False
    
    def test_delete_document_success(self, mock_store, sample_documents):
        """测试成功删除文档"""
        mock_store.add_documents(sample_documents)
        
        result = mock_store.delete_document("doc1")
        
        assert result == True
        assert mock_store.get_document_count() == 1
        assert mock_store.get_document_by_id("doc1") is None
    
    def test_delete_document_not_exists(self, mock_store):
        """测试删除不存在的文档"""
        result = mock_store.delete_document("nonexistent")
        
        assert result == False
    
    def test_delete_all_documents(self, mock_store, sample_documents):
        """测试删除所有文档"""
        mock_store.add_documents(sample_documents)
        assert mock_store.get_document_count() == 2
        
        result = mock_store.delete_all_documents()
        
        assert result == True
        assert mock_store.get_document_count() == 0
    
    def test_get_document_count(self, mock_store, sample_documents):
        """测试获取文档数量"""
        assert mock_store.get_document_count() == 0
        
        mock_store.add_documents(sample_documents)
        assert mock_store.get_document_count() == 2
        
        mock_store.delete_document("doc1")
        assert mock_store.get_document_count() == 1


class TestVectorStoreExceptions:
    """向量存储异常测试"""
    
    def test_vector_store_error(self):
        """测试基础异常"""
        with pytest.raises(VectorStoreError):
            raise VectorStoreError("Test error")
    
    def test_vector_store_connection_error(self):
        """测试连接异常"""
        with pytest.raises(VectorStoreConnectionError):
            raise VectorStoreConnectionError("Connection failed")
    
    def test_vector_store_operation_error(self):
        """测试操作异常"""
        with pytest.raises(VectorStoreOperationError):
            raise VectorStoreOperationError("Operation failed")
    
    def test_vector_store_config_error(self):
        """测试配置异常"""
        with pytest.raises(VectorStoreConfigError):
            raise VectorStoreConfigError("Config error")
    
    def test_exception_inheritance(self):
        """测试异常继承关系"""
        assert issubclass(VectorStoreConnectionError, VectorStoreError)
        assert issubclass(VectorStoreOperationError, VectorStoreError)  
        assert issubclass(VectorStoreConfigError, VectorStoreError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])