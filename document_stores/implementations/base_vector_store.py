"""
统一向量存储接口基类

该模块定义了向量存储系统的抽象基类，提供统一的接口规范，
支持FAISS、Milvus等不同的向量存储后端的无缝切换。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from haystack import Document as HaystackDocument


class BaseVectorStore(ABC):
    """
    向量存储的抽象基类。
    
    定义了所有向量存储后端必须实现的核心接口，确保不同存储引擎
    之间的一致性和可互换性。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化向量存储基类。
        
        Args:
            config: 存储配置字典
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vector_dim = config.get("vector_dimension", 384)
        self.similarity_function = config.get("similarity", "dot_product")
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化向量存储后端。
        
        Returns:
            初始化成功返回True，失败返回False
        """
        pass
    
    @abstractmethod
    def add_documents(
        self, 
        documents: List[HaystackDocument],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """
        批量添加文档到向量存储。
        
        Args:
            documents: Haystack文档列表
            embeddings: 对应的向量嵌入列表，可选
            
        Returns:
            添加成功返回True，失败返回False
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[HaystackDocument, float]]:
        """
        向量相似度搜索。
        
        Args:
            query_embedding: 查询向量
            top_k: 返回的最相似文档数量
            filters: 元数据过滤条件，可选
            
        Returns:
            包含(文档, 相似度得分)的元组列表
        """
        pass
    
    @abstractmethod
    def get_document_by_id(self, doc_id: str) -> Optional[HaystackDocument]:
        """
        根据文档ID获取文档。
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档对象，不存在则返回None
        """
        pass
    
    @abstractmethod
    def update_document(
        self, 
        doc_id: str,
        document: HaystackDocument,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        更新文档。
        
        Args:
            doc_id: 文档ID
            document: 新的文档对象
            embedding: 新的向量嵌入，可选
            
        Returns:
            更新成功返回True，失败返回False
        """
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档。
        
        Args:
            doc_id: 文档ID
            
        Returns:
            删除成功返回True，失败返回False
        """
        pass
    
    @abstractmethod
    def delete_all_documents(self) -> bool:
        """
        删除所有文档。
        
        Returns:
            删除成功返回True，失败返回False
        """
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """
        获取文档总数。
        
        Returns:
            文档总数
        """
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> bool:
        """
        保存索引到文件。
        
        Args:
            path: 保存路径
            
        Returns:
            保存成功返回True，失败返回False
        """
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> bool:
        """
        从文件加载索引。
        
        Args:
            path: 索引文件路径
            
        Returns:
            加载成功返回True，失败返回False
        """
        pass
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        获取存储信息。
        
        Returns:
            包含存储类型、配置等信息的字典
        """
        return {
            "storage_type": self.__class__.__name__,
            "vector_dimension": self.vector_dim,
            "similarity_function": self.similarity_function,
            "is_initialized": self.is_initialized,
            "document_count": self.get_document_count() if self.is_initialized else 0
        }
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        验证向量嵌入的有效性。
        
        Args:
            embedding: 向量嵌入
            
        Returns:
            有效返回True，无效返回False
        """
        if not embedding:
            return False
        
        if len(embedding) != self.vector_dim:
            self.logger.warning(f"向量维度不匹配: 期望{self.vector_dim}, 实际{len(embedding)}")
            return False
        
        # 检查是否包含非法值
        try:
            import math
            for val in embedding:
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    self.logger.warning("向量包含非法数值")
                    return False
        except Exception as e:
            self.logger.error(f"验证向量时发生错误: {str(e)}")
            return False
        
        return True
    
    def get_supported_similarity_functions(self) -> List[str]:
        """
        获取支持的相似度函数列表。
        
        Returns:
            相似度函数名称列表
        """
        return ["dot_product", "cosine", "euclidean"]
    
    def close(self):
        """
        关闭存储连接，清理资源。
        
        子类可以重写此方法以实现特定的资源清理逻辑。
        """
        self.is_initialized = False
        self.logger.info(f"{self.__class__.__name__} 已关闭")


class VectorStoreError(Exception):
    """向量存储相关异常类"""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """向量存储连接异常"""
    pass


class VectorStoreOperationError(VectorStoreError):
    """向量存储操作异常"""
    pass


class VectorStoreConfigError(VectorStoreError):
    """向量存储配置异常"""
    pass