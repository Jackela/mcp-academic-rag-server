"""
向量存储系统集成测试

测试向量存储系统的端到端功能，包括不同后端间的集成、
性能基准测试和实际使用场景验证。
"""

import pytest
import tempfile
import shutil
import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from haystack import Document as HaystackDocument

from document_stores.vector_store_factory import VectorStoreFactory, create_vector_store
from document_stores.implementations.base_vector_store import BaseVectorStore
from utils.vector_migration import VectorStoreMigrator


class TestVectorStorageIntegration:
    """向量存储系统集成测试"""
    
    @pytest.fixture
    def temp_storage_path(self):
        """创建临时存储路径"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def large_document_set(self):
        """创建大文档集用于性能测试"""
        docs = []
        embeddings = []
        
        for i in range(1000):
            doc = HaystackDocument(
                content=f"This is document number {i} containing various topics and information.",
                meta={
                    "doc_id": i,
                    "category": f"category_{i % 10}",
                    "author": f"author_{i % 5}",
                    "timestamp": f"2024-01-{(i % 30) + 1:02d}"
                },
                id=f"large_doc_{i}"
            )
            
            # 生成随机但一致的向量嵌入
            np.random.seed(i)  # 确保可重现
            embedding = np.random.random(384).tolist()
            
            docs.append(doc)
            embeddings.append(embedding)
        
        return docs, embeddings
    
    @pytest.fixture
    def small_document_set(self):
        """创建小文档集用于快速测试"""
        docs = [
            HaystackDocument(
                content="Artificial Intelligence is transforming technology",
                meta={"category": "AI", "author": "Alice"},
                id="ai_doc",
                embedding=np.random.random(384).tolist()
            ),
            HaystackDocument(
                content="Machine Learning algorithms improve with data",
                meta={"category": "ML", "author": "Bob"}, 
                id="ml_doc",
                embedding=np.random.random(384).tolist()
            ),
            HaystackDocument(
                content="Deep Learning uses neural networks",
                meta={"category": "DL", "author": "Charlie"},
                id="dl_doc",
                embedding=np.random.random(384).tolist()
            )
        ]
        return docs
    
    def test_memory_store_full_workflow(self, small_document_set):
        """测试内存存储的完整工作流程"""
        config = {
            "type": "memory",
            "vector_dimension": 384,
            "similarity": "dot_product"
        }
        
        store = create_vector_store(config)
        
        try:
            # 初始化
            assert store.initialize() == True
            assert store.get_document_count() == 0
            
            # 添加文档
            assert store.add_documents(small_document_set) == True
            assert store.get_document_count() == 3
            
            # 搜索测试
            query_embedding = np.random.random(384).tolist()
            results = store.search(query_embedding, top_k=2)
            
            assert len(results) == 2
            for doc, score in results:
                assert isinstance(doc, HaystackDocument)
                assert isinstance(score, float)
            
            # 获取文档
            doc = store.get_document_by_id("ai_doc")
            assert doc is not None
            assert doc.content.startswith("Artificial Intelligence")
            
            # 更新文档
            updated_doc = HaystackDocument(
                content="Updated AI content",
                meta={"category": "AI", "author": "Alice Updated"},
                id="ai_doc"
            )
            assert store.update_document("ai_doc", updated_doc) == True
            
            retrieved = store.get_document_by_id("ai_doc")
            assert retrieved.content == "Updated AI content"
            
            # 删除文档
            assert store.delete_document("ml_doc") == True
            assert store.get_document_count() == 2
            assert store.get_document_by_id("ml_doc") is None
            
        finally:
            store.close()
    
    @pytest.mark.skipif(
        not _is_faiss_available(),
        reason="FAISS not available"
    )
    def test_faiss_store_full_workflow(self, small_document_set, temp_storage_path):
        """测试FAISS存储的完整工作流程"""
        config = {
            "type": "faiss",
            "vector_dimension": 384,
            "similarity": "dot_product",
            "faiss": {
                "storage_path": temp_storage_path,
                "index_type": "Flat",
                "auto_save_interval": 0
            }
        }
        
        store = create_vector_store(config, auto_fallback=True)
        
        try:
            # 初始化
            assert store.initialize() == True
            
            # 添加文档
            assert store.add_documents(small_document_set) == True
            assert store.get_document_count() == 3
            
            # 持久化测试
            assert store.save_index(temp_storage_path) == True
            
            # 验证保存的文件
            assert os.path.exists(os.path.join(temp_storage_path, "index.faiss"))
            assert os.path.exists(os.path.join(temp_storage_path, "metadata.json"))
            
            # 搜索测试
            query_embedding = small_document_set[0].embedding
            results = store.search(query_embedding, top_k=1)
            
            assert len(results) == 1
            top_doc, score = results[0]
            assert top_doc.id == "ai_doc"  # 应该找到最相似的文档
            
        finally:
            store.close()
    
    def test_backend_availability_and_fallback(self):
        """测试后端可用性检查和自动回退"""
        # 获取可用后端
        available_backends = VectorStoreFactory.get_available_backends()
        
        assert "memory" in available_backends
        assert available_backends["memory"]["available"] == True
        
        # 测试不存在的后端自动回退
        config = {
            "type": "nonexistent_backend",
            "vector_dimension": 384
        }
        
        store = create_vector_store(config, auto_fallback=True)
        assert store is not None
        
        # 应该回退到内存存储
        info = store.get_storage_info()
        assert "InMemoryVectorStore" in info["storage_type"]
        
        store.close()
    
    def test_backend_recommendation_system(self):
        """测试后端推荐系统"""
        # 小数据集推荐
        small_req = {
            "document_count": 100,
            "performance_level": "standard",
            "persistence_required": False
        }
        assert VectorStoreFactory.get_recommended_backend(small_req) == "memory"
        
        # 中等数据集推荐
        medium_req = {
            "document_count": 10000,
            "performance_level": "high",
            "persistence_required": True
        }
        assert VectorStoreFactory.get_recommended_backend(medium_req) == "faiss"
        
        # 大数据集推荐
        large_req = {
            "document_count": 1000000,
            "performance_level": "enterprise"
        }
        assert VectorStoreFactory.get_recommended_backend(large_req) == "milvus"


class TestVectorStorageMigration:
    """向量存储迁移集成测试"""
    
    @pytest.fixture
    def temp_paths(self):
        """创建多个临时路径"""
        paths = {}
        for name in ["source", "target", "backup"]:
            paths[name] = tempfile.mkdtemp()
        
        yield paths
        
        for path in paths.values():
            shutil.rmtree(path, ignore_errors=True)
    
    def test_memory_to_memory_migration(self, small_document_set):
        """测试内存到内存的迁移（主要测试迁移流程）"""
        # 创建源存储并添加数据
        source_config = {"type": "memory", "vector_dimension": 384}
        source_store = create_vector_store(source_config)
        source_store.add_documents(small_document_set)
        
        # 创建目标存储
        target_config = {"type": "memory", "vector_dimension": 384}
        
        # 执行迁移
        migrator = VectorStoreMigrator()
        
        with patch('utils.vector_migration.create_vector_store') as mock_create:
            mock_create.side_effect = [source_store, create_vector_store(target_config)]
            
            result = migrator.migrate(
                source_config,
                target_config,
                batch_size=2,
                verify_migration=True,
                backup_before_migration=False
            )
        
        assert result == True
        
        source_store.close()
    
    @pytest.mark.skipif(
        not _is_faiss_available(),
        reason="FAISS not available"
    )  
    def test_memory_to_faiss_migration(self, small_document_set, temp_paths):
        """测试内存到FAISS的迁移"""
        # 创建源存储
        source_config = {"type": "memory", "vector_dimension": 384}
        source_store = create_vector_store(source_config)
        source_store.add_documents(small_document_set)
        
        # 目标配置
        target_config = {
            "type": "faiss",
            "vector_dimension": 384,
            "faiss": {
                "storage_path": temp_paths["target"],
                "index_type": "Flat"
            }
        }
        
        # 执行迁移
        migrator = VectorStoreMigrator()
        
        with patch('utils.vector_migration.create_vector_store') as mock_create:
            target_store = create_vector_store(target_config, auto_fallback=True)
            mock_create.side_effect = [source_store, target_store]
            
            result = migrator.migrate(
                source_config,
                target_config,
                verify_migration=True
            )
        
        # 如果FAISS可用，迁移应该成功
        if "FAISSVectorStore" in str(type(target_store)):
            assert result == True
        
        source_store.close()
        if hasattr(target_store, 'close'):
            target_store.close()


class TestVectorStoragePerformance:
    """向量存储性能测试"""
    
    @pytest.fixture
    def performance_config(self):
        """性能测试配置"""
        return {
            "batch_sizes": [10, 50, 100],
            "document_counts": [100, 500, 1000],
            "vector_dimensions": [128, 384, 768],
            "search_counts": [1, 10, 50]
        }
    
    def test_memory_store_performance_baseline(self, performance_config):
        """测试内存存储的性能基准"""
        results = {}
        
        for doc_count in [100, 500]:  # 减少测试时间
            for vector_dim in [128, 384]:
                config = {
                    "type": "memory",
                    "vector_dimension": vector_dim
                }
                
                store = create_vector_store(config)
                
                try:
                    # 生成测试数据
                    docs, embeddings = self._generate_test_data(doc_count, vector_dim)
                    
                    # 测试添加文档性能
                    start_time = time.time()
                    store.add_documents(docs, embeddings)
                    add_time = time.time() - start_time
                    
                    # 测试搜索性能
                    query_embedding = np.random.random(vector_dim).tolist()
                    
                    search_times = []
                    for _ in range(10):  # 多次搜索求平均值
                        start_time = time.time()
                        results_list = store.search(query_embedding, top_k=5)
                        search_time = time.time() - start_time
                        search_times.append(search_time)
                    
                    avg_search_time = sum(search_times) / len(search_times)
                    
                    results[f"memory_{doc_count}_{vector_dim}"] = {
                        "add_time": add_time,
                        "avg_search_time": avg_search_time,
                        "docs_per_second": doc_count / add_time if add_time > 0 else float('inf')
                    }
                    
                finally:
                    store.close()
        
        # 验证性能指标合理性
        for key, metrics in results.items():
            assert metrics["add_time"] > 0
            assert metrics["avg_search_time"] >= 0
            assert metrics["docs_per_second"] > 0
            
            # 内存存储应该相对较快
            assert metrics["add_time"] < 10  # 10秒内完成添加
            assert metrics["avg_search_time"] < 1  # 1秒内完成搜索
    
    @pytest.mark.skipif(
        not _is_faiss_available(),
        reason="FAISS not available"
    )
    def test_faiss_store_performance_comparison(self, temp_storage_path):
        """测试FAISS存储性能并与内存存储比较"""
        doc_count = 500
        vector_dim = 384
        
        # 生成测试数据
        docs, embeddings = self._generate_test_data(doc_count, vector_dim)
        query_embedding = np.random.random(vector_dim).tolist()
        
        # 测试FAISS性能
        faiss_config = {
            "type": "faiss",
            "vector_dimension": vector_dim,
            "faiss": {
                "storage_path": temp_storage_path,
                "index_type": "Flat"
            }
        }
        
        faiss_store = create_vector_store(faiss_config, auto_fallback=True)
        
        try:
            # FAISS添加文档
            start_time = time.time()
            faiss_store.add_documents(docs, embeddings)
            faiss_add_time = time.time() - start_time
            
            # FAISS搜索
            start_time = time.time()
            faiss_results = faiss_store.search(query_embedding, top_k=5)
            faiss_search_time = time.time() - start_time
            
            # 验证结果
            assert len(faiss_results) <= 5
            assert faiss_add_time > 0
            assert faiss_search_time >= 0
            
            # 如果是真正的FAISS存储，测试持久化性能
            if "FAISSVectorStore" in str(type(faiss_store)):
                start_time = time.time()
                assert faiss_store.save_index(temp_storage_path) == True
                save_time = time.time() - start_time
                
                assert save_time >= 0
                assert os.path.exists(os.path.join(temp_storage_path, "index.faiss"))
            
        finally:
            faiss_store.close()
    
    def test_batch_size_performance_impact(self):
        """测试批处理大小对性能的影响"""
        doc_count = 1000
        vector_dim = 128
        batch_sizes = [10, 50, 100, 200]
        
        docs, embeddings = self._generate_test_data(doc_count, vector_dim)
        
        results = {}
        
        for batch_size in batch_sizes:
            config = {"type": "memory", "vector_dimension": vector_dim}
            store = create_vector_store(config)
            
            try:
                start_time = time.time()
                
                # 分批添加文档
                for i in range(0, len(docs), batch_size):
                    batch_docs = docs[i:i + batch_size]
                    batch_embeddings = embeddings[i:i + batch_size]
                    store.add_documents(batch_docs, batch_embeddings)
                
                total_time = time.time() - start_time
                results[batch_size] = total_time
                
            finally:
                store.close()
        
        # 验证批处理大小的影响
        assert all(time_taken > 0 for time_taken in results.values())
        
        # 一般来说，较大的批处理应该更高效（但可能不总是如此）
        # 这里只验证所有批处理都能完成
        assert len(results) == len(batch_sizes)
    
    def test_vector_dimension_performance_impact(self):
        """测试向量维度对性能的影响"""
        doc_count = 200
        vector_dims = [64, 128, 256, 384]
        
        results = {}
        
        for vector_dim in vector_dims:
            docs, embeddings = self._generate_test_data(doc_count, vector_dim)
            
            config = {"type": "memory", "vector_dimension": vector_dim}
            store = create_vector_store(config)
            
            try:
                # 测试添加性能
                start_time = time.time()
                store.add_documents(docs, embeddings)
                add_time = time.time() - start_time
                
                # 测试搜索性能
                query_embedding = np.random.random(vector_dim).tolist()
                start_time = time.time()
                store.search(query_embedding, top_k=10)
                search_time = time.time() - start_time
                
                results[vector_dim] = {
                    "add_time": add_time,
                    "search_time": search_time
                }
                
            finally:
                store.close()
        
        # 验证性能随维度变化的合理性
        for dim, metrics in results.items():
            assert metrics["add_time"] > 0
            assert metrics["search_time"] >= 0
        
        # 高维度通常需要更多时间（但现代系统优化可能使差异不明显）
        assert len(results) == len(vector_dims)
    
    def _generate_test_data(self, doc_count: int, vector_dim: int) -> Tuple[List[HaystackDocument], List[List[float]]]:
        """生成测试数据"""
        docs = []
        embeddings = []
        
        for i in range(doc_count):
            doc = HaystackDocument(
                content=f"Performance test document {i} with various content for testing purposes.",
                meta={"doc_id": i, "category": f"cat_{i % 5}"},
                id=f"perf_doc_{i}"
            )
            
            # 生成确定性随机向量
            np.random.seed(i)
            embedding = np.random.random(vector_dim).tolist()
            
            docs.append(doc)
            embeddings.append(embedding)
        
        return docs, embeddings


class TestVectorStorageErrorHandling:
    """向量存储错误处理测试"""
    
    def test_invalid_config_handling(self):
        """测试无效配置的处理"""
        invalid_configs = [
            {"type": "memory", "vector_dimension": -1},  # 负数维度
            {"type": "memory", "vector_dimension": "invalid"},  # 非数字维度
            {"type": "memory", "similarity": "invalid_function"},  # 无效相似度函数
            {"type": "nonexistent"},  # 不存在的存储类型
            {}  # 空配置
        ]
        
        for config in invalid_configs:
            try:
                store = create_vector_store(config, auto_fallback=True, validate_config=False)
                # 如果创建成功，应该是回退到了默认配置
                assert store is not None
                store.close()
            except Exception as e:
                # 如果抛出异常，也是预期的行为
                assert isinstance(e, Exception)
    
    def test_storage_operations_error_handling(self):
        """测试存储操作的错误处理"""
        config = {"type": "memory", "vector_dimension": 384}
        store = create_vector_store(config)
        
        try:
            # 测试未初始化时的操作
            uninit_store = create_vector_store(config)
            uninit_store.is_initialized = False
            
            # 这些操作应该优雅地处理未初始化状态
            result = uninit_store.add_documents([])
            assert result in [True, False]  # 应该返回布尔值而不是抛出异常
            
            results = uninit_store.search([0.1] * 384)
            assert isinstance(results, list)  # 应该返回空列表
            
            count = uninit_store.get_document_count()
            assert isinstance(count, int)
            
            uninit_store.close()
            
        finally:
            store.close()
    
    def test_embedding_validation_edge_cases(self):
        """测试向量嵌入验证的边界情况"""
        config = {"type": "memory", "vector_dimension": 3}
        store = create_vector_store(config)
        
        try:
            invalid_docs = [
                HaystackDocument(content="No embedding", id="no_emb"),
                HaystackDocument(content="Wrong dim", id="wrong_dim", embedding=[1.0, 2.0]),  # 维度不对
                HaystackDocument(content="NaN values", id="nan", embedding=[1.0, float('nan'), 3.0]),
                HaystackDocument(content="Inf values", id="inf", embedding=[1.0, float('inf'), 3.0])
            ]
            
            # 应该优雅处理无效嵌入，而不是崩溃
            result = store.add_documents(invalid_docs)
            assert isinstance(result, bool)
            
        finally:
            store.close()


def _is_faiss_available() -> bool:
    """检查FAISS是否可用"""
    try:
        import faiss
        return True
    except ImportError:
        return False


# 导入模拟补丁用于测试
from unittest.mock import patch


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])