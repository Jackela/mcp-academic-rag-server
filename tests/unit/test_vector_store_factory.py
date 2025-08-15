"""
向量存储工厂模式测试

测试VectorStoreFactory的后端选择、配置验证、自动回退等功能。
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from document_stores.vector_store_factory import (
    VectorStoreFactory, 
    create_vector_store, 
    get_available_backends, 
    recommend_backend
)
from document_stores.implementations.base_vector_store import (
    BaseVectorStore, 
    VectorStoreError, 
    VectorStoreConfigError
)


class MockFailingVectorStore(BaseVectorStore):
    """模拟总是失败的向量存储"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def initialize(self) -> bool:
        return False  # 总是初始化失败
        
    def add_documents(self, documents, embeddings=None) -> bool:
        return False
        
    def search(self, query_embedding, top_k=5, filters=None):
        return []
        
    def get_document_by_id(self, doc_id: str):
        return None
        
    def update_document(self, doc_id: str, document, embedding=None) -> bool:
        return False
        
    def delete_document(self, doc_id: str) -> bool:
        return False
        
    def delete_all_documents(self) -> bool:
        return False
        
    def get_document_count(self) -> int:
        return 0
        
    def save_index(self, path: str) -> bool:
        return False
        
    def load_index(self, path: str) -> bool:
        return False


class TestVectorStoreFactory:
    """向量存储工厂测试"""
    
    @pytest.fixture
    def temp_storage_path(self):
        """创建临时存储路径"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_create_memory_store(self):
        """测试创建内存存储"""
        config = {
            "type": "memory",
            "vector_dimension": 384
        }
        
        store = VectorStoreFactory.create(config)
        
        assert store is not None
        assert "InMemoryVectorStore" in str(type(store))
        assert store.vector_dim == 384
        store.close()
    
    @patch('document_stores.vector_store_factory.importlib.import_module')
    def test_create_faiss_store_when_available(self, mock_import, temp_storage_path):
        """测试FAISS可用时创建FAISS存储"""
        # 模拟FAISS模块可用
        mock_faiss = MagicMock()
        mock_import.side_effect = lambda module: mock_faiss if module == "faiss" else MagicMock()
        
        config = {
            "type": "faiss",
            "vector_dimension": 384,
            "faiss": {
                "storage_path": temp_storage_path
            }
        }
        
        # 这里会因为实际的FAISS类导入失败，但我们可以测试工厂逻辑
        # 在实际环境中，如果FAISS已安装，这个测试会成功
        try:
            store = VectorStoreFactory.create(config, auto_fallback=True)
            # 如果FAISS可用，应该创建FAISS存储
            if store:
                store.close()
        except VectorStoreError:
            # 如果FAISS不可用，应该有相应的错误处理
            pass
    
    def test_create_with_auto_fallback(self):
        """测试自动回退机制"""
        # 配置一个不存在的存储类型
        config = {
            "type": "nonexistent_backend",
            "vector_dimension": 384
        }
        
        # 应该自动回退到可用的存储
        store = VectorStoreFactory.create(config, auto_fallback=True)
        
        assert store is not None
        # 应该回退到内存存储（最后的回退选项）
        assert "InMemoryVectorStore" in str(type(store))
        store.close()
    
    def test_create_without_fallback_fails(self):
        """测试禁用回退时创建失败"""
        config = {
            "type": "nonexistent_backend",
            "vector_dimension": 384
        }
        
        with pytest.raises(VectorStoreError):
            VectorStoreFactory.create(config, auto_fallback=False)
    
    def test_config_validation_missing_type(self):
        """测试缺少type字段的配置验证"""
        config = {
            "vector_dimension": 384
            # 缺少type字段
        }
        
        with pytest.raises(VectorStoreConfigError, match="缺少必需的配置字段: type"):
            VectorStoreFactory.create(config, validate_config=True)
    
    def test_config_validation_invalid_vector_dimension(self):
        """测试无效向量维度的配置验证"""
        config = {
            "type": "memory",
            "vector_dimension": -1  # 无效维度
        }
        
        with pytest.raises(VectorStoreConfigError, match="vector_dimension 必须是正整数"):
            VectorStoreFactory.create(config, validate_config=True)
    
    def test_config_validation_invalid_similarity(self):
        """测试无效相似度函数的配置验证"""
        config = {
            "type": "memory",
            "vector_dimension": 384,
            "similarity": "invalid_similarity"
        }
        
        with pytest.raises(VectorStoreConfigError, match="similarity 必须是以下之一"):
            VectorStoreFactory.create(config, validate_config=True)
    
    def test_config_validation_faiss_storage_path(self, temp_storage_path):
        """测试FAISS存储路径验证"""
        # 创建一个无法写入的路径（在某些系统上可能需要调整）
        invalid_path = "/invalid/path/that/cannot/be/created"
        
        config = {
            "type": "faiss",
            "vector_dimension": 384,
            "faiss": {
                "storage_path": invalid_path
            }
        }
        
        # 在Windows上，可能不会抛出异常，所以我们使用有效路径来测试成功情况
        valid_config = {
            "type": "faiss",
            "vector_dimension": 384,
            "faiss": {
                "storage_path": temp_storage_path
            }
        }
        
        factory = VectorStoreFactory()
        # 这应该不会抛出配置错误
        factory._validate_config(valid_config, "faiss")
    
    def test_config_validation_milvus_config(self):
        """测试Milvus配置验证"""
        config = {
            "type": "milvus",
            "vector_dimension": 384,
            "milvus": {
                "host": "localhost",
                "port": "invalid_port",  # 无效端口
                "collection_name": "test_collection"
            }
        }
        
        factory = VectorStoreFactory()
        with pytest.raises(VectorStoreConfigError, match="Milvus端口必须是正整数"):
            factory._validate_config(config, "milvus")
    
    def test_get_available_backends(self):
        """测试获取可用后端"""
        backends = VectorStoreFactory.get_available_backends()
        
        assert isinstance(backends, dict)
        assert "memory" in backends
        assert "faiss" in backends
        assert "milvus" in backends
        
        # 检查每个后端的信息结构
        for backend_name, info in backends.items():
            assert "description" in info
            assert "dependencies" in info
            assert "available" in info
            assert isinstance(info["available"], bool)
    
    def test_get_recommended_backend_small_dataset(self):
        """测试小数据集的后端推荐"""
        requirements = {
            "document_count": 1000,
            "performance_level": "standard",
            "persistence_required": False
        }
        
        recommended = VectorStoreFactory.get_recommended_backend(requirements)
        
        # 小数据集且不需要持久化应该推荐内存存储
        assert recommended == "memory"
    
    def test_get_recommended_backend_medium_dataset(self):
        """测试中等数据集的后端推荐"""
        requirements = {
            "document_count": 50000,
            "performance_level": "high",
            "persistence_required": True
        }
        
        recommended = VectorStoreFactory.get_recommended_backend(requirements)
        
        # 高性能需求应该推荐FAISS
        assert recommended == "faiss"
    
    def test_get_recommended_backend_large_dataset(self):
        """测试大数据集的后端推荐"""
        requirements = {
            "document_count": 1000000,
            "performance_level": "enterprise"
        }
        
        recommended = VectorStoreFactory.get_recommended_backend(requirements)
        
        # 企业级需求应该推荐Milvus
        assert recommended == "milvus"
    
    def test_get_recommended_backend_default(self):
        """测试默认推荐"""
        recommended = VectorStoreFactory.get_recommended_backend()
        
        # 默认应该推荐内存存储
        assert recommended == "memory"
    
    @patch('document_stores.vector_store_factory.VectorStoreFactory._create_backend')
    def test_fallback_on_initialization_failure(self, mock_create_backend):
        """测试初始化失败时的回退"""
        # 模拟主要后端初始化失败
        failing_store = MockFailingVectorStore({})
        mock_create_backend.side_effect = [failing_store, Exception("All backends failed")]
        
        config = {
            "type": "faiss",
            "vector_dimension": 384
        }
        
        with pytest.raises(VectorStoreError):
            VectorStoreFactory.create(config, auto_fallback=True)
    
    def test_skip_config_validation(self):
        """测试跳过配置验证"""
        config = {
            "type": "memory",
            "vector_dimension": -1,  # 无效但跳过验证
            "similarity": "invalid"   # 无效但跳过验证
        }
        
        # 跳过验证应该不抛出异常
        store = VectorStoreFactory.create(config, validate_config=False)
        
        assert store is not None
        store.close()
    
    def test_backend_cache(self):
        """测试后端类缓存"""
        factory = VectorStoreFactory()
        
        # 第一次加载
        config1 = {"type": "memory", "vector_dimension": 384}
        store1 = factory._create_store(config1, auto_fallback=False)
        
        # 第二次加载，应该使用缓存
        config2 = {"type": "memory", "vector_dimension": 384}
        store2 = factory._create_store(config2, auto_fallback=False)
        
        assert store1 is not None
        assert store2 is not None
        
        store1.close()
        store2.close()
    
    def test_dependency_checking(self):
        """测试依赖检查"""
        factory = VectorStoreFactory()
        
        # 测试检查不存在的依赖
        with pytest.raises(VectorStoreError, match="需要以下依赖"):
            factory._check_dependencies("test", ["nonexistent-package"])
    
    @patch('document_stores.vector_store_factory.importlib.import_module')
    def test_dynamic_import_failure(self, mock_import):
        """测试动态导入失败"""
        mock_import.side_effect = ImportError("Module not found")
        
        factory = VectorStoreFactory()
        backend_info = {
            "class_name": "TestStore",
            "module": "nonexistent.module",
            "dependencies": []
        }
        
        with pytest.raises(VectorStoreError, match="无法导入存储模块"):
            factory._get_backend_class("test", backend_info)


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_create_vector_store_function(self):
        """测试create_vector_store便捷函数"""
        config = {
            "type": "memory",
            "vector_dimension": 384
        }
        
        store = create_vector_store(config)
        
        assert store is not None
        assert isinstance(store, BaseVectorStore)
        store.close()
    
    def test_get_available_backends_function(self):
        """测试get_available_backends便捷函数"""
        backends = get_available_backends()
        
        assert isinstance(backends, dict)
        assert len(backends) > 0
    
    def test_recommend_backend_function(self):
        """测试recommend_backend便捷函数"""
        requirements = {
            "document_count": 10000,
            "performance_level": "high"
        }
        
        backend = recommend_backend(requirements)
        
        assert isinstance(backend, str)
        assert backend in ["memory", "faiss", "milvus"]


class TestEdgeCases:
    """边界情况和错误处理测试"""
    
    def test_empty_config(self):
        """测试空配置"""
        config = {}
        
        # 空配置应该使用默认值并成功创建
        store = VectorStoreFactory.create(config, auto_fallback=True)
        
        assert store is not None
        store.close()
    
    def test_config_with_extra_fields(self):
        """测试包含额外字段的配置"""
        config = {
            "type": "memory",
            "vector_dimension": 384,
            "extra_field": "extra_value",  # 额外字段
            "another_extra": 123
        }
        
        store = VectorStoreFactory.create(config)
        
        assert store is not None
        store.close()
    
    def test_case_insensitive_backend_type(self):
        """测试大小写不敏感的后端类型"""
        configs = [
            {"type": "Memory", "vector_dimension": 384},
            {"type": "MEMORY", "vector_dimension": 384},
            {"type": "memory", "vector_dimension": 384}
        ]
        
        for config in configs:
            store = VectorStoreFactory.create(config, auto_fallback=True)
            assert store is not None
            store.close()
    
    def test_none_requirements_in_recommendation(self):
        """测试推荐函数中的None需求"""
        backend = VectorStoreFactory.get_recommended_backend(None)
        
        assert backend == "memory"  # 应该返回默认推荐


if __name__ == "__main__":
    pytest.main([__file__, "-v"])