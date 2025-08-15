"""
Document Stores Package

提供多种向量存储后端的统一接口和实现。

模块结构:
- base_vector_store: 抽象基类定义
- implementations/: 具体存储后端实现
  - faiss_vector_store: FAISS高性能向量存储
  - milvus_store: Milvus分布式向量存储  
  - memory_vector_store: 内存向量存储
  - haystack_store: Haystack文档存储适配器
- migration/: 数据迁移和备份工具
- vector_store_factory: 统一工厂类
"""

from .implementations.base_vector_store import BaseVectorStore, VectorStoreError, VectorStoreConnectionError
from .vector_store_factory import VectorStoreFactory
from .implementations.faiss_vector_store import FAISSVectorStore
from .implementations.memory_vector_store import InMemoryVectorStore
try:
    from .implementations.milvus_store import MilvusDocumentStore
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    MilvusDocumentStore = None
from .implementations.haystack_store import HaystackDocumentStore
from .migration.vector_migration import VectorStoreMigrator

__all__ = [
    "BaseVectorStore",
    "VectorStoreError", 
    "VectorStoreConnectionError",
    "VectorStoreFactory",
    "FAISSVectorStore",
    "InMemoryVectorStore", 
    "HaystackDocumentStore",
    "VectorStoreMigrator"
]

# Add MilvusDocumentStore to __all__ only if available
if MILVUS_AVAILABLE:
    __all__.append("MilvusDocumentStore")