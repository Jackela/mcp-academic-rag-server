"""
FAISS向量存储测试

测试FAISSVectorStore的功能、性能和持久化特性。
"""

import pytest
import tempfile
import os
import shutil
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from haystack import Document as HaystackDocument

from document_stores.implementations.base_vector_store import VectorStoreConnectionError
from document_stores.faiss_vector_store import FAISSVectorStore

# 模拟FAISS不可用的情况
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
class TestFAISSVectorStore:
    """FAISS向量存储功能测试"""
    
    @pytest.fixture
    def temp_storage_path(self):
        """创建临时存储路径"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def basic_config(self, temp_storage_path):
        """基础FAISS配置"""
        return {
            "vector_dimension": 4,
            "similarity": "dot_product",
            "faiss": {
                "storage_path": temp_storage_path,
                "index_type": "Flat",
                "auto_save_interval": 0  # 禁用自动保存以便测试
            }
        }
    
    @pytest.fixture
    def sample_documents(self):
        """创建示例文档"""
        docs = [
            HaystackDocument(
                content="Document about artificial intelligence",
                meta={"category": "AI", "author": "Alice"},
                id="doc1",
                embedding=[1.0, 0.0, 0.0, 0.0]
            ),
            HaystackDocument(
                content="Document about machine learning",
                meta={"category": "ML", "author": "Bob"}, 
                id="doc2",
                embedding=[0.0, 1.0, 0.0, 0.0]
            ),
            HaystackDocument(
                content="Document about deep learning",
                meta={"category": "DL", "author": "Charlie"},
                id="doc3", 
                embedding=[0.0, 0.0, 1.0, 0.0]
            )
        ]
        return docs
    
    def test_initialization_success(self, basic_config):
        """测试成功初始化"""
        store = FAISSVectorStore(basic_config)
        
        assert store.vector_dim == 4
        assert store.similarity_function == "dot_product"
        assert store.index_type == "Flat"
        assert not store.is_initialized
        
        result = store.initialize()
        
        assert result == True
        assert store.is_initialized
        assert store.index is not None
    
    def test_initialization_with_different_index_types(self, temp_storage_path):
        """测试不同索引类型的初始化"""
        index_types = ["Flat", "IVF100,Flat", "HNSW"]
        
        for index_type in index_types:
            config = {
                "vector_dimension": 8,
                "faiss": {
                    "storage_path": temp_storage_path,
                    "index_type": index_type,
                    "index_params": {"M": 8, "efConstruction": 40} if "HNSW" in index_type else {}
                }
            }
            
            store = FAISSVectorStore(config)
            result = store.initialize()
            
            assert result == True, f"Failed to initialize {index_type}"
            store.close()
    
    def test_initialization_creates_storage_directory(self, temp_storage_path):
        """测试初始化时创建存储目录"""
        storage_path = os.path.join(temp_storage_path, "new_directory")
        
        config = {
            "vector_dimension": 4,
            "faiss": {
                "storage_path": storage_path,
                "index_type": "Flat"
            }
        }
        
        store = FAISSVectorStore(config)
        
        assert os.path.exists(storage_path)
    
    def test_add_documents_success(self, basic_config, sample_documents):
        """测试成功添加文档"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        
        result = store.add_documents(sample_documents)
        
        assert result == True
        assert store.get_document_count() == 3
        
        # 验证文档映射
        assert "doc1" in store.documents
        assert "doc2" in store.documents
        assert "doc3" in store.documents
        
        store.close()
    
    def test_add_documents_with_external_embeddings(self, basic_config, sample_documents):
        """测试使用外部嵌入添加文档"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        
        # 移除文档自带的嵌入
        for doc in sample_documents:
            doc.embedding = None
        
        external_embeddings = [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0, 1.0]
        ]
        
        result = store.add_documents(sample_documents, external_embeddings)
        
        assert result == True
        assert store.get_document_count() == 3
        
        store.close()
    
    def test_add_documents_invalid_embeddings(self, basic_config):
        """测试添加无效嵌入的文档"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        
        invalid_doc = HaystackDocument(
            content="Document with invalid embedding",
            id="invalid_doc",
            embedding=[1.0, 2.0]  # 错误维度
        )
        
        result = store.add_documents([invalid_doc])
        
        assert result == False
        assert store.get_document_count() == 0
        
        store.close()
    
    def test_search_documents(self, basic_config, sample_documents):
        """测试文档搜索"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        store.add_documents(sample_documents)
        
        # 搜索与第一个文档相似的内容
        query_embedding = [1.0, 0.1, 0.0, 0.0]
        results = store.search(query_embedding, top_k=2)
        
        assert len(results) == 2
        
        # 验证返回格式
        for doc, score in results:
            assert isinstance(doc, HaystackDocument)
            assert isinstance(score, float)
        
        # 最相似的应该是doc1
        top_doc, top_score = results[0]
        assert top_doc.id == "doc1"
        
        store.close()
    
    def test_search_with_filters(self, basic_config, sample_documents):
        """测试带过滤器的搜索"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        store.add_documents(sample_documents)
        
        query_embedding = [0.5, 0.5, 0.5, 0.5]
        filters = {"category": "AI"}
        
        results = store.search(query_embedding, top_k=5, filters=filters)
        
        # 应该只返回category为AI的文档
        for doc, score in results:
            assert doc.meta.get("category") == "AI"
        
        store.close()
    
    def test_get_document_by_id(self, basic_config, sample_documents):
        """测试通过ID获取文档"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        store.add_documents(sample_documents)
        
        doc = store.get_document_by_id("doc2")
        
        assert doc is not None
        assert doc.id == "doc2"
        assert doc.content == "Document about machine learning"
        
        # 测试不存在的ID
        non_existent = store.get_document_by_id("nonexistent")
        assert non_existent is None
        
        store.close()
    
    def test_update_document(self, basic_config, sample_documents):
        """测试更新文档"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        store.add_documents(sample_documents)
        
        updated_doc = HaystackDocument(
            content="Updated document about AI",
            meta={"category": "AI", "author": "Alice Updated"},
            id="doc1",
            embedding=[0.9, 0.1, 0.0, 0.0]
        )
        
        result = store.update_document("doc1", updated_doc, [0.9, 0.1, 0.0, 0.0])
        
        assert result == True
        
        # 验证更新
        retrieved = store.get_document_by_id("doc1")
        assert retrieved.content == "Updated document about AI"
        assert retrieved.meta["author"] == "Alice Updated"
        
        store.close()
    
    def test_delete_document(self, basic_config, sample_documents):
        """测试删除文档"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        store.add_documents(sample_documents)
        
        initial_count = store.get_document_count()
        
        result = store.delete_document("doc2")
        
        assert result == True
        assert store.get_document_count() == initial_count - 1
        assert store.get_document_by_id("doc2") is None
        
        # 测试删除不存在的文档
        result = store.delete_document("nonexistent")
        assert result == False
        
        store.close()
    
    def test_delete_all_documents(self, basic_config, sample_documents):
        """测试删除所有文档"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        store.add_documents(sample_documents)
        
        assert store.get_document_count() > 0
        
        result = store.delete_all_documents()
        
        assert result == True
        assert store.get_document_count() == 0
        
        store.close()
    
    def test_save_and_load_index(self, basic_config, sample_documents, temp_storage_path):
        """测试索引保存和加载"""
        # 创建存储并添加文档
        store1 = FAISSVectorStore(basic_config)
        store1.initialize()
        store1.add_documents(sample_documents)
        
        # 保存索引
        save_result = store1.save_index(temp_storage_path)
        assert save_result == True
        
        # 验证保存的文件存在
        assert os.path.exists(os.path.join(temp_storage_path, "index.faiss"))
        assert os.path.exists(os.path.join(temp_storage_path, "metadata.json"))
        
        store1.close()
        
        # 创建新存储并加载索引
        store2 = FAISSVectorStore(basic_config)
        store2.initialize()
        
        load_result = store2.load_index(temp_storage_path)
        assert load_result == True
        
        # 验证加载的数据
        assert store2.get_document_count() == 3
        
        doc1 = store2.get_document_by_id("doc1")
        assert doc1 is not None
        assert doc1.content == "Document about artificial intelligence"
        
        store2.close()
    
    def test_cosine_similarity(self, temp_storage_path, sample_documents):
        """测试余弦相似度"""
        config = {
            "vector_dimension": 4,
            "similarity": "cosine",
            "faiss": {
                "storage_path": temp_storage_path,
                "index_type": "Flat"
            }
        }
        
        store = FAISSVectorStore(config)
        store.initialize()
        store.add_documents(sample_documents)
        
        # 查询与第一个文档相同的向量（应该有最高相似度）
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results = store.search(query_embedding, top_k=1)
        
        assert len(results) == 1
        doc, score = results[0]
        assert doc.id == "doc1"
        # 余弦相似度应该接近1.0
        assert score > 0.9
        
        store.close()
    
    def test_ivf_index_training(self, temp_storage_path):
        """测试IVF索引训练"""
        config = {
            "vector_dimension": 4,
            "faiss": {
                "storage_path": temp_storage_path,
                "index_type": "IVF10,Flat"
            }
        }
        
        store = FAISSVectorStore(config)
        store.initialize()
        
        # 创建足够多的文档用于训练IVF索引
        docs = []
        embeddings = []
        for i in range(300):  # IVF需要至少256个向量
            doc = HaystackDocument(
                content=f"Document {i}",
                id=f"doc_{i}"
            )
            docs.append(doc)
            # 生成随机向量
            embedding = np.random.random(4).tolist()
            embeddings.append(embedding)
        
        result = store.add_documents(docs, embeddings)
        
        assert result == True
        assert store.get_document_count() == 300
        assert store.index.is_trained == True
        
        store.close()
    
    def test_get_index_stats(self, basic_config, sample_documents):
        """测试获取索引统计信息"""
        store = FAISSVectorStore(basic_config)
        store.initialize()
        store.add_documents(sample_documents)
        
        stats = store.get_index_stats()
        
        assert stats["total_vectors"] == 3
        assert stats["vector_dimension"] == 4
        assert stats["index_type"] == "Flat"
        assert stats["is_trained"] == True
        assert stats["use_gpu"] == False
        
        store.close()
    
    @patch('faiss.StandardGpuResources')
    @patch('faiss.index_cpu_to_gpu')
    def test_gpu_support(self, mock_cpu_to_gpu, mock_gpu_resources, temp_storage_path):
        """测试GPU支持（模拟）"""
        config = {
            "vector_dimension": 4,
            "faiss": {
                "storage_path": temp_storage_path,
                "index_type": "Flat",
                "use_gpu": True
            }
        }
        
        # 模拟GPU资源和转换
        mock_gpu_resources.return_value = MagicMock()
        mock_cpu_to_gpu.return_value = MagicMock()
        
        store = FAISSVectorStore(config)
        result = store.initialize()
        
        assert result == True
        assert mock_gpu_resources.called
        assert mock_cpu_to_gpu.called
        
        store.close()


@pytest.mark.skipif(FAISS_AVAILABLE, reason="Testing FAISS unavailable scenario")
class TestFAISSNotAvailable:
    """测试FAISS不可用时的行为"""
    
    def test_faiss_not_available_error(self):
        """测试FAISS不可用时抛出异常"""
        config = {"vector_dimension": 4, "faiss": {}}
        
        with pytest.raises(VectorStoreConnectionError, match="FAISS未安装"):
            FAISSVectorStore(config)


class TestFAISSConfigValidation:
    """FAISS配置验证测试"""
    
    @pytest.fixture
    def temp_storage_path(self):
        """创建临时存储路径"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_invalid_index_type(self, temp_storage_path):
        """测试无效的索引类型"""
        config = {
            "vector_dimension": 4,
            "faiss": {
                "storage_path": temp_storage_path,
                "index_type": "InvalidType"
            }
        }
        
        store = FAISSVectorStore(config)
        result = store.initialize()
        
        # 应该回退到Flat索引
        assert result == True
        assert store.index is not None
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_default_config_values(self, temp_storage_path):
        """测试默认配置值"""
        config = {
            "vector_dimension": 4,
            "faiss": {
                "storage_path": temp_storage_path
            }
        }
        
        store = FAISSVectorStore(config)
        
        assert store.index_type == "Flat"
        assert store.metric_type == "INNER_PRODUCT"
        assert store.auto_save_interval == 300
        assert store.use_gpu == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])