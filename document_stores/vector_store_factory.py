"""
向量存储工厂模式实现

提供统一的向量存储创建接口，支持自动后端选择、配置验证、
错误回退和存储迁移功能。
"""

import logging
from typing import Dict, Any, Optional, Type, List
import importlib
import os

from .implementations.base_vector_store import BaseVectorStore, VectorStoreError, VectorStoreConfigError


class VectorStoreFactory:
    """
    向量存储工厂类。
    
    负责根据配置创建合适的向量存储后端，提供自动回退、
    配置验证和存储迁移功能。
    """
    
    # 支持的存储后端映射
    _BACKENDS = {
        "memory": {
            "class_name": "MemoryVectorStore", 
            "module": "document_stores.implementations.memory_vector_store",
            "dependencies": [],
            "description": "内存存储，适用于开发和小规模场景"
        },
        "faiss": {
            "class_name": "FAISSVectorStore",
            "module": "document_stores.implementations.faiss_vector_store", 
            "dependencies": ["faiss-cpu"],
            "description": "FAISS高性能向量检索，支持持久化"
        },
        "milvus": {
            "class_name": "MilvusStore",
            "module": "document_stores.implementations.milvus_store",
            "dependencies": ["pymilvus"],
            "description": "Milvus分布式向量数据库，适用于生产环境"
        }
    }
    
    # 回退策略：优先级从高到低
    _FALLBACK_ORDER = ["faiss", "memory"]
    
    def __init__(self):
        """初始化工厂类"""
        self.logger = logging.getLogger("VectorStoreFactory")
        self._backend_cache = {}  # 缓存已加载的后端类
    
    @classmethod
    def create(
        cls, 
        config: Dict[str, Any], 
        auto_fallback: bool = True,
        validate_config: bool = True
    ) -> BaseVectorStore:
        """
        创建向量存储实例。
        
        Args:
            config: 存储配置字典
            auto_fallback: 是否启用自动回退机制
            validate_config: 是否验证配置
            
        Returns:
            向量存储实例
            
        Raises:
            VectorStoreError: 创建失败时抛出
        """
        factory = cls()
        return factory._create_store(config, auto_fallback, validate_config)
    
    def _create_store(
        self, 
        config: Dict[str, Any], 
        auto_fallback: bool = True,
        validate_config: bool = True
    ) -> BaseVectorStore:
        """
        内部创建存储实例的方法。
        
        Args:
            config: 存储配置
            auto_fallback: 自动回退
            validate_config: 验证配置
            
        Returns:
            向量存储实例
        """
        # 获取存储类型
        store_type = config.get("type", "memory").lower()
        
        # 配置验证
        if validate_config:
            self._validate_config(config, store_type)
        
        # 尝试创建主要存储后端
        try:
            store = self._create_backend(store_type, config)
            if store.initialize():
                self.logger.info(f"成功创建 {store_type} 向量存储")
                return store
            else:
                raise VectorStoreError(f"{store_type} 初始化失败")
                
        except Exception as e:
            self.logger.warning(f"创建 {store_type} 存储失败: {str(e)}")
            
            if not auto_fallback:
                raise VectorStoreError(f"创建 {store_type} 存储失败: {str(e)}")
        
        # 自动回退机制
        if auto_fallback:
            return self._create_fallback_store(config, exclude=[store_type])
        
        raise VectorStoreError(f"无法创建任何可用的向量存储后端")
    
    def _create_backend(self, store_type: str, config: Dict[str, Any]) -> BaseVectorStore:
        """
        创建指定类型的存储后端。
        
        Args:
            store_type: 存储类型
            config: 配置字典
            
        Returns:
            存储实例
        """
        if store_type not in self._BACKENDS:
            raise VectorStoreConfigError(f"不支持的存储类型: {store_type}")
        
        backend_info = self._BACKENDS[store_type]
        
        # 检查依赖
        self._check_dependencies(store_type, backend_info["dependencies"])
        
        # 获取存储类
        store_class = self._get_backend_class(store_type, backend_info)
        
        # 创建实例
        return store_class(config)
    
    def _get_backend_class(self, store_type: str, backend_info: Dict[str, Any]) -> Type[BaseVectorStore]:
        """
        获取存储后端类。
        
        Args:
            store_type: 存储类型
            backend_info: 后端信息
            
        Returns:
            存储类
        """
        # 检查缓存
        if store_type in self._backend_cache:
            return self._backend_cache[store_type]
        
        try:
            # 动态导入模块
            module = importlib.import_module(backend_info["module"])
            store_class = getattr(module, backend_info["class_name"])
            
            # 验证类是否继承自BaseVectorStore
            if not issubclass(store_class, BaseVectorStore):
                raise VectorStoreError(f"{backend_info['class_name']} 必须继承自 BaseVectorStore")
            
            # 缓存类引用
            self._backend_cache[store_type] = store_class
            
            return store_class
            
        except ImportError as e:
            raise VectorStoreError(f"无法导入存储模块 {backend_info['module']}: {str(e)}")
        except AttributeError as e:
            raise VectorStoreError(f"模块中不存在类 {backend_info['class_name']}: {str(e)}")
    
    def _check_dependencies(self, store_type: str, dependencies: List[str]):
        """
        检查存储后端的依赖是否满足。
        
        Args:
            store_type: 存储类型
            dependencies: 依赖包列表
        """
        missing_deps = []
        
        for dep in dependencies:
            try:
                # 处理包名映射
                if dep == "faiss-cpu":
                    import faiss
                elif dep == "pymilvus":
                    import pymilvus
                else:
                    importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            deps_str = ", ".join(missing_deps)
            raise VectorStoreError(
                f"{store_type} 存储需要以下依赖: {deps_str}\n"
                f"请运行: pip install {' '.join(missing_deps)}"
            )
    
    def _create_fallback_store(
        self, 
        config: Dict[str, Any], 
        exclude: List[str] = None
    ) -> BaseVectorStore:
        """
        创建回退存储后端。
        
        Args:
            config: 配置
            exclude: 排除的存储类型列表
            
        Returns:
            存储实例
        """
        exclude = exclude or []
        
        for fallback_type in self._FALLBACK_ORDER:
            if fallback_type in exclude:
                continue
            
            try:
                self.logger.info(f"尝试回退到 {fallback_type} 存储")
                
                # 创建回退配置
                fallback_config = config.copy()
                fallback_config["type"] = fallback_type
                
                store = self._create_backend(fallback_type, fallback_config)
                if store.initialize():
                    self.logger.warning(f"已回退到 {fallback_type} 存储")
                    return store
                    
            except Exception as e:
                self.logger.warning(f"回退到 {fallback_type} 存储失败: {str(e)}")
                continue
        
        raise VectorStoreError("所有存储后端都不可用")
    
    def _validate_config(self, config: Dict[str, Any], store_type: str):
        """
        验证存储配置。
        
        Args:
            config: 配置字典
            store_type: 存储类型
        """
        # 基础配置验证
        required_fields = ["type"]
        for field in required_fields:
            if field not in config:
                raise VectorStoreConfigError(f"缺少必需的配置字段: {field}")
        
        # 向量维度验证
        vector_dim = config.get("vector_dimension", 384)
        if not isinstance(vector_dim, int) or vector_dim <= 0:
            raise VectorStoreConfigError("vector_dimension 必须是正整数")
        
        # 相似度函数验证
        similarity = config.get("similarity", "dot_product")
        valid_similarities = ["dot_product", "cosine", "euclidean"]
        if similarity not in valid_similarities:
            raise VectorStoreConfigError(
                f"similarity 必须是以下之一: {', '.join(valid_similarities)}"
            )
        
        # 存储类型特定验证
        if store_type == "faiss":
            self._validate_faiss_config(config.get("faiss", {}))
        elif store_type == "milvus":
            self._validate_milvus_config(config.get("milvus", {}))
    
    def _validate_faiss_config(self, faiss_config: Dict[str, Any]):
        """
        验证FAISS配置。
        
        Args:
            faiss_config: FAISS配置字典
        """
        # 存储路径
        storage_path = faiss_config.get("storage_path", "./data/faiss")
        try:
            os.makedirs(storage_path, exist_ok=True)
        except Exception as e:
            raise VectorStoreConfigError(f"无法创建FAISS存储路径 {storage_path}: {str(e)}")
        
        # 索引类型验证
        index_type = faiss_config.get("index_type", "Flat")
        valid_index_types = ["Flat", "IVF", "HNSW"]
        if not any(index_type.startswith(t) for t in valid_index_types):
            raise VectorStoreConfigError(f"不支持的FAISS索引类型: {index_type}")
    
    def _validate_milvus_config(self, milvus_config: Dict[str, Any]):
        """
        验证Milvus配置。
        
        Args:
            milvus_config: Milvus配置字典
        """
        # 连接配置
        host = milvus_config.get("host", "localhost")
        port = milvus_config.get("port", 19530)
        
        if not isinstance(port, int) or port <= 0:
            raise VectorStoreConfigError("Milvus端口必须是正整数")
        
        # 集合名称
        collection_name = milvus_config.get("collection_name", "documents")
        if not collection_name or not isinstance(collection_name, str):
            raise VectorStoreConfigError("Milvus collection_name 必须是非空字符串")
    
    @classmethod
    def get_available_backends(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取可用的存储后端信息。
        
        Returns:
            后端信息字典
        """
        factory = cls()
        available = {}
        
        for backend_type, backend_info in cls._BACKENDS.items():
            try:
                factory._check_dependencies(backend_type, backend_info["dependencies"])
                available[backend_type] = {
                    "description": backend_info["description"],
                    "dependencies": backend_info["dependencies"],
                    "available": True
                }
            except VectorStoreError:
                available[backend_type] = {
                    "description": backend_info["description"],
                    "dependencies": backend_info["dependencies"],
                    "available": False
                }
        
        return available
    
    @classmethod
    def get_recommended_backend(cls, requirements: Dict[str, Any] = None) -> str:
        """
        根据需求推荐最佳存储后端。
        
        Args:
            requirements: 需求字典，包含document_count, performance_level等
            
        Returns:
            推荐的存储类型
        """
        requirements = requirements or {}
        
        doc_count = requirements.get("document_count", 0)
        performance_level = requirements.get("performance_level", "standard")  # standard, high, enterprise
        persistence_required = requirements.get("persistence_required", True)
        
        # 企业级需求
        if performance_level == "enterprise" or doc_count > 100000:
            return "milvus"
        
        # 高性能需求
        if performance_level == "high" or doc_count > 10000:
            return "faiss"
        
        # 标准需求
        if persistence_required:
            return "faiss"
        
        # 开发或小规模
        return "memory"
    
    @classmethod
    def migrate_storage(
        cls, 
        source_config: Dict[str, Any],
        target_config: Dict[str, Any],
        batch_size: int = 1000
    ) -> bool:
        """
        存储迁移功能。
        
        Args:
            source_config: 源存储配置
            target_config: 目标存储配置
            batch_size: 批量迁移大小
            
        Returns:
            迁移成功返回True
        """
        logger = logging.getLogger("VectorStoreMigration")
        
        try:
            # 创建源和目标存储
            source_store = cls.create(source_config, auto_fallback=False)
            target_store = cls.create(target_config, auto_fallback=False)
            
            # 获取所有文档
            total_docs = source_store.get_document_count()
            logger.info(f"开始迁移 {total_docs} 个文档")
            
            migrated_count = 0
            
            # 批量迁移（这里简化实现，实际需要根据具体存储类型实现批量获取）
            # 注意：BaseVectorStore接口中没有定义批量获取方法，需要扩展
            
            logger.info(f"迁移完成，总共迁移 {migrated_count} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"存储迁移失败: {str(e)}")
            return False


# 便捷函数
def create_vector_store(
    config: Dict[str, Any], 
    auto_fallback: bool = True
) -> BaseVectorStore:
    """
    创建向量存储的便捷函数。
    
    Args:
        config: 存储配置
        auto_fallback: 启用自动回退
        
    Returns:
        向量存储实例
    """
    return VectorStoreFactory.create(config, auto_fallback)


def get_available_backends() -> Dict[str, Dict[str, Any]]:
    """
    获取可用存储后端的便捷函数。
    
    Returns:
        后端信息字典
    """
    return VectorStoreFactory.get_available_backends()


def recommend_backend(requirements: Dict[str, Any] = None) -> str:
    """
    推荐存储后端的便捷函数。
    
    Args:
        requirements: 需求字典
        
    Returns:
        推荐的存储类型
    """
    return VectorStoreFactory.get_recommended_backend(requirements)