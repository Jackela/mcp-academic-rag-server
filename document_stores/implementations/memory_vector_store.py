"""
内存向量存储实现

基于Haystack InMemoryDocumentStore的向量存储适配器，
提供与现有系统的向后兼容性。
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document as HaystackDocument

from .base_vector_store import BaseVectorStore, VectorStoreError


class InMemoryVectorStore(BaseVectorStore):
    """
    基于Haystack InMemoryDocumentStore的内存向量存储。
    
    提供与现有系统的完全兼容性，适用于开发、测试和小规模场景。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化内存向量存储。
        
        Args:
            config: 存储配置字典
        """
        super().__init__(config)
        
        # Haystack配置映射
        similarity_map = {
            "dot_product": "dot_product",
            "cosine": "cosine", 
            "euclidean": "dot_product"  # InMemoryDocumentStore不直接支持欧氏距离
        }
        
        haystack_similarity = similarity_map.get(self.similarity_function, "dot_product")
        
        # 创建Haystack文档存储
        self.store = InMemoryDocumentStore(
            embedding_similarity_function=haystack_similarity,
            return_embedding=True
        )
        
        self.logger.info("InMemoryVectorStore初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化内存存储（无需特殊初始化）。
        
        Returns:
            总是返回True
        """
        self.is_initialized = True
        self.logger.info("内存向量存储初始化成功")
        return True
    
    def add_documents(
        self, 
        documents: List[HaystackDocument],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """
        添加文档到内存存储。
        
        Args:
            documents: Haystack文档列表
            embeddings: 对应的向量嵌入列表
            
        Returns:
            添加成功返回True，失败返回False
        """
        if not self.is_initialized:
            self.logger.error("存储未初始化")
            return False
        
        if not documents:
            return True
        
        try:
            # 处理向量嵌入
            docs_to_add = []
            for i, doc in enumerate(documents):
                # 复制文档以避免修改原始对象
                new_doc = HaystackDocument(
                    content=doc.content,
                    meta=doc.meta.copy() if doc.meta else {},
                    id=doc.id
                )
                
                # 设置向量嵌入
                if embeddings and i < len(embeddings):
                    embedding = embeddings[i]
                    if self.validate_embedding(embedding):
                        new_doc.embedding = embedding
                    else:
                        self.logger.warning(f"文档 {doc.id} 的向量嵌入无效，跳过")
                        continue
                elif doc.embedding:
                    if self.validate_embedding(doc.embedding):
                        new_doc.embedding = doc.embedding
                    else:
                        self.logger.warning(f"文档 {doc.id} 的向量嵌入无效，跳过")
                        continue
                else:
                    self.logger.warning(f"文档 {doc.id} 缺少向量嵌入，跳过")
                    continue
                
                docs_to_add.append(new_doc)
            
            if docs_to_add:
                self.store.write_documents(docs_to_add)
                self.logger.info(f"成功添加 {len(docs_to_add)} 个文档到内存存储")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"添加文档失败: {str(e)}")
            return False
    
    def search(
        self, 
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[HaystackDocument, float]]:
        """
        在内存存储中进行向量搜索。
        
        Args:
            query_embedding: 查询向量
            top_k: 返回的最相似文档数量
            filters: 元数据过滤条件
            
        Returns:
            包含(文档, 相似度得分)的元组列表
        """
        if not self.is_initialized:
            self.logger.error("存储未初始化")
            return []
        
        if not self.validate_embedding(query_embedding):
            self.logger.error("查询向量无效")
            return []
        
        try:
            # Haystack过滤器格式转换
            haystack_filters = None
            if filters:
                haystack_filters = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        haystack_filters[key] = {"$in": value}
                    else:
                        haystack_filters[key] = value
            
            # 执行搜索
            results = self.store.embedding_retrieval(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=haystack_filters
            )
            
            # 格式化结果
            search_results = []
            for doc in results:
                # 计算相似度得分（Haystack可能没有直接提供得分）
                if hasattr(doc, 'score') and doc.score is not None:
                    score = float(doc.score)
                else:
                    # 手动计算相似度
                    if doc.embedding:
                        score = self._calculate_similarity(query_embedding, doc.embedding)
                    else:
                        score = 0.0
                
                search_results.append((doc, score))
            
            # 按得分排序
            search_results.sort(key=lambda x: x[1], reverse=True)
            
            return search_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"搜索失败: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[HaystackDocument]:
        """
        根据文档ID获取文档。
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档对象，不存在则返回None
        """
        try:
            docs = self.store.get_documents_by_id([doc_id])
            return docs[0] if docs else None
        except Exception as e:
            self.logger.error(f"获取文档失败: {str(e)}")
            return None
    
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
            embedding: 新的向量嵌入
            
        Returns:
            更新成功返回True，失败返回False
        """
        try:
            # 删除旧文档
            self.store.delete_documents([doc_id])
            
            # 添加新文档
            document.id = doc_id
            if embedding:
                document.embedding = embedding
            
            return self.add_documents([document])
            
        except Exception as e:
            self.logger.error(f"更新文档失败: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档。
        
        Args:
            doc_id: 文档ID
            
        Returns:
            删除成功返回True，失败返回False
        """
        try:
            self.store.delete_documents([doc_id])
            self.logger.info(f"文档 {doc_id} 已删除")
            return True
        except Exception as e:
            self.logger.error(f"删除文档失败: {str(e)}")
            return False
    
    def delete_all_documents(self) -> bool:
        """
        删除所有文档。
        
        Returns:
            删除成功返回True，失败返回False
        """
        try:
            self.store.delete_documents()
            self.logger.info("所有文档已清空")
            return True
        except Exception as e:
            self.logger.error(f"清空文档失败: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """
        获取文档总数。
        
        Returns:
            文档总数
        """
        try:
            return self.store.get_document_count()
        except Exception as e:
            self.logger.error(f"获取文档数量失败: {str(e)}")
            return 0
    
    def save_index(self, path: str) -> bool:
        """
        保存索引（内存存储不支持持久化）。
        
        Args:
            path: 保存路径
            
        Returns:
            总是返回False（不支持持久化）
        """
        self.logger.warning("InMemoryVectorStore不支持持久化存储")
        return False
    
    def load_index(self, path: str) -> bool:
        """
        加载索引（内存存储不支持持久化）。
        
        Args:
            path: 索引文件路径
            
        Returns:
            总是返回False（不支持持久化）
        """
        self.logger.warning("InMemoryVectorStore不支持从文件加载")
        return False
    
    def get_haystack_store(self) -> InMemoryDocumentStore:
        """
        获取底层Haystack存储对象（用于兼容性）。
        
        Returns:
            InMemoryDocumentStore对象
        """
        return self.store
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的相似度。
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            相似度得分
        """
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            if self.similarity_function == "cosine":
                # 余弦相似度
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return float(np.dot(v1, v2) / (norm1 * norm2))
            
            elif self.similarity_function == "euclidean":
                # 欧氏距离转相似度
                distance = np.linalg.norm(v1 - v2)
                return float(1.0 / (1.0 + distance))
            
            else:  # dot_product
                return float(np.dot(v1, v2))
                
        except Exception as e:
            self.logger.error(f"计算相似度失败: {str(e)}")
            return 0.0