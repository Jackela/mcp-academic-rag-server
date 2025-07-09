"""
基于Haystack框架的文档存储实现，主要使用InMemoryDocumentStore。

该模块提供了HaystackDocumentStore类，封装了Haystack的InMemoryDocumentStore，
用于存储文档及其向量表示，并支持未来迁移到其他存储引擎。
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json

from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document as HaystackDocument

from models.document import Document
from core.config_manager import ConfigManager


class HaystackDocumentStore:
    """
    基于Haystack框架的文档存储类，主要使用InMemoryDocumentStore。
    
    该类封装了Haystack的InMemoryDocumentStore功能，用于存储文档及其向量表示，
    并提供统一的接口供系统其他组件使用。它支持文档的添加、检索、更新和删除，
    并可通过配置文件调整参数。
    
    设计为可扩展接口，支持未来迁移到其他存储引擎（如FAISS、Elasticsearch等）。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_manager: Optional[ConfigManager] = None):
        """
        初始化HaystackDocumentStore对象。
        
        Args:
            config: 文档存储配置字典，默认为None
            config_manager: 配置管理器对象，默认为None
        """
        self.logger = logging.getLogger("haystack_document_store")
        
        # 优先使用传入的配置，如果未提供则尝试从配置管理器获取
        self.config = config or {}
        if config_manager:
            self.config = config_manager.get_value("vector_db.document_store", {})
        
        # 配置参数
        self.embedding_dim = self.config.get("embedding_dim", 768)  # 默认使用768维向量
        self.similarity = self.config.get("similarity", "dot_product")  # 默认使用点积相似度
        self.return_embedding = self.config.get("return_embedding", True)
        
        # 初始化Haystack文档存储
        self.document_store = self._initialize_document_store()
        
        # 记录文档ID映射，用于跟踪系统Document对象与Haystack Document的对应关系
        self.id_mapping = {}
        
        self.logger.info("HaystackDocumentStore初始化完成")
    
    def _initialize_document_store(self) -> InMemoryDocumentStore:
        """
        初始化Haystack文档存储引擎。
        
        Returns:
            初始化好的InMemoryDocumentStore对象
        """
        try:
            store = InMemoryDocumentStore(
                embedding_dim=self.embedding_dim,
                similarity=self.similarity,
                return_embedding=self.return_embedding
            )
            self.logger.info(f"成功初始化InMemoryDocumentStore，embedding_dim={self.embedding_dim}, similarity={self.similarity}")
            return store
        except Exception as e:
            self.logger.error(f"初始化InMemoryDocumentStore失败: {str(e)}")
            # 创建一个基本配置的存储作为备用
            return InMemoryDocumentStore(embedding_dim=768)
    
    def add_document(self, document: Document, embedding: Optional[List[float]] = None) -> bool:
        """
        将系统Document对象添加到Haystack文档存储中。
        
        Args:
            document: 系统Document对象
            embedding: 文档的向量表示，默认为None
            
        Returns:
            如果成功添加则返回True，否则返回False
        """
        try:
            # 构建Haystack文档内容，优先使用嵌入处理器生成的内容
            text = document.get_content("EmbeddingProcessor") or document.get_content("OCRProcessor") or ""
            
            # 如果没有文本内容，则返回失败
            if not text:
                self.logger.warning(f"文档 {document.document_id} 没有可用的文本内容，无法添加到文档存储")
                return False
            
            # 创建Haystack文档对象
            haystack_doc = HaystackDocument(
                content=text,
                meta={
                    "file_path": document.file_path,
                    "file_name": document.file_name,
                    "file_type": document.file_type,
                    "original_id": document.document_id,
                    "metadata": document.metadata,
                    "tags": document.tags
                },
                embedding=embedding
            )
            
            # 将Haystack文档添加到存储
            self.document_store.write_documents([haystack_doc])
            
            # 记录ID映射关系
            self.id_mapping[document.document_id] = haystack_doc.id
            
            self.logger.info(f"文档 {document.document_id} 已成功添加到Haystack文档存储")
            return True
            
        except Exception as e:
            self.logger.error(f"添加文档 {document.document_id} 到Haystack文档存储失败: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> Tuple[int, int]:
        """
        批量将系统Document对象添加到Haystack文档存储中。
        
        Args:
            documents: 系统Document对象列表
            embeddings: 文档向量表示列表，默认为None
            
        Returns:
            包含成功数量和失败数量的元组
        """
        success_count = 0
        fail_count = 0
        
        # 如果提供了嵌入向量，确保数量匹配
        if embeddings and len(documents) != len(embeddings):
            self.logger.error("文档数量与嵌入向量数量不匹配")
            return 0, len(documents)
        
        # 批量处理文档
        haystack_docs = []
        
        for i, document in enumerate(documents):
            try:
                # 获取文档内容
                text = document.get_content("EmbeddingProcessor") or document.get_content("OCRProcessor") or ""
                
                if not text:
                    self.logger.warning(f"文档 {document.document_id} 没有可用的文本内容，跳过")
                    fail_count += 1
                    continue
                
                # 获取对应的嵌入向量
                embedding = embeddings[i] if embeddings else None
                
                # 创建Haystack文档对象
                haystack_doc = HaystackDocument(
                    content=text,
                    meta={
                        "file_path": document.file_path,
                        "file_name": document.file_name,
                        "file_type": document.file_type,
                        "original_id": document.document_id,
                        "metadata": document.metadata,
                        "tags": document.tags
                    },
                    embedding=embedding
                )
                
                haystack_docs.append(haystack_doc)
                # 记录ID映射关系
                self.id_mapping[document.document_id] = haystack_doc.id
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"处理文档 {document.document_id} 失败: {str(e)}")
                fail_count += 1
        
        # 将文档批量写入存储
        if haystack_docs:
            try:
                self.document_store.write_documents(haystack_docs)
                self.logger.info(f"成功批量添加 {len(haystack_docs)} 个文档到Haystack文档存储")
            except Exception as e:
                self.logger.error(f"批量写入文档失败: {str(e)}")
                return 0, len(documents)
        
        return success_count, fail_count
    
    def get_document(self, document_id: str) -> Optional[HaystackDocument]:
        """
        根据文档ID获取Haystack文档。
        
        Args:
            document_id: 系统文档ID
            
        Returns:
            Haystack文档对象，如果不存在则返回None
        """
        try:
            haystack_id = self.id_mapping.get(document_id)
            if not haystack_id:
                self.logger.warning(f"未找到文档ID {document_id} 的映射")
                return None
            
            docs = self.document_store.get_documents_by_id([haystack_id])
            if docs:
                return docs[0]
            return None
            
        except Exception as e:
            self.logger.error(f"获取文档 {document_id} 失败: {str(e)}")
            return None
    
    def update_document(self, document: Document, embedding: Optional[List[float]] = None) -> bool:
        """
        更新Haystack文档存储中的文档。
        
        Args:
            document: 系统Document对象
            embedding: 文档的新向量表示，默认为None
            
        Returns:
            如果成功更新则返回True，否则返回False
        """
        try:
            # 首先检查文档是否存在
            haystack_id = self.id_mapping.get(document.document_id)
            if not haystack_id:
                self.logger.warning(f"文档 {document.document_id} 不存在，无法更新")
                return False
            
            # 删除原有文档
            self.document_store.delete_documents([haystack_id])
            
            # 添加更新后的文档
            return self.add_document(document, embedding)
            
        except Exception as e:
            self.logger.error(f"更新文档 {document.document_id} 失败: {str(e)}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        从Haystack文档存储中删除文档。
        
        Args:
            document_id: 系统文档ID
            
        Returns:
            如果成功删除则返回True，否则返回False
        """
        try:
            haystack_id = self.id_mapping.get(document_id)
            if not haystack_id:
                self.logger.warning(f"文档 {document_id} 不存在，无法删除")
                return False
            
            # 删除文档
            self.document_store.delete_documents([haystack_id])
            
            # 移除ID映射
            del self.id_mapping[document_id]
            
            self.logger.info(f"文档 {document_id} 已成功从Haystack文档存储中删除")
            return True
            
        except Exception as e:
            self.logger.error(f"删除文档 {document_id} 失败: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        清空Haystack文档存储中的所有文档。
        
        Returns:
            如果成功清空则返回True，否则返回False
        """
        try:
            self.document_store.delete_all_documents()
            self.id_mapping.clear()
            self.logger.info("已清空Haystack文档存储")
            return True
        except Exception as e:
            self.logger.error(f"清空Haystack文档存储失败: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """
        获取Haystack文档存储中的文档数量。
        
        Returns:
            文档数量
        """
        try:
            return self.document_store.get_document_count()
        except Exception as e:
            self.logger.error(f"获取文档数量失败: {str(e)}")
            return 0
    
    def save_state(self, file_path: str) -> bool:
        """
        保存文档存储状态到文件，便于恢复。
        
        Args:
            file_path: 状态文件路径
            
        Returns:
            如果成功保存则返回True，否则返回False
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 将ID映射保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.id_mapping, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"文档存储状态已保存到 {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存文档存储状态失败: {str(e)}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """
        从文件加载文档存储状态。
        
        Args:
            file_path: 状态文件路径
            
        Returns:
            如果成功加载则返回True，否则返回False
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"状态文件 {file_path} 不存在")
                return False
            
            # 从文件加载ID映射
            with open(file_path, 'r', encoding='utf-8') as f:
                self.id_mapping = json.load(f)
            
            self.logger.info(f"已从 {file_path} 加载文档存储状态")
            return True
        except Exception as e:
            self.logger.error(f"加载文档存储状态失败: {str(e)}")
            return False
    
    def get_document_store(self) -> InMemoryDocumentStore:
        """
        获取底层Haystack文档存储对象。
        
        Returns:
            Haystack InMemoryDocumentStore对象
        """
        return self.document_store
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新文档存储配置。
        
        部分配置项更新后需要重新初始化文档存储。
        
        Args:
            config: 新的配置字典
            
        Returns:
            如果成功更新则返回True，否则返回False
        """
        try:
            # 保存当前文档，用于重新初始化后恢复
            current_docs = self.document_store.get_all_documents()
            
            # 更新配置
            self.config.update(config)
            
            # 更新核心参数
            if "embedding_dim" in config:
                self.embedding_dim = config["embedding_dim"]
            if "similarity" in config:
                self.similarity = config["similarity"]
            if "return_embedding" in config:
                self.return_embedding = config["return_embedding"]
            
            # 重新初始化文档存储
            self.document_store = self._initialize_document_store()
            
            # 恢复文档
            if current_docs:
                self.document_store.write_documents(current_docs)
            
            self.logger.info("已更新文档存储配置并重新初始化")
            return True
        except Exception as e:
            self.logger.error(f"更新文档存储配置失败: {str(e)}")
            return False


# 工厂函数，便于创建文档存储实例
def create_document_store(config: Optional[Dict[str, Any]] = None, config_manager: Optional[ConfigManager] = None) -> HaystackDocumentStore:
    """
    创建文档存储实例的工厂函数。
    
    Args:
        config: 文档存储配置字典，默认为None
        config_manager: 配置管理器对象，默认为None
        
    Returns:
        HaystackDocumentStore实例
    """
    return HaystackDocumentStore(config, config_manager)
