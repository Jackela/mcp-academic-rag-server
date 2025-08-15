"""
FAISS向量存储实现

基于Facebook AI Similarity Search (FAISS) 的高性能向量存储实现，
支持持久化存储、索引优化和增量更新。
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from haystack import Document as HaystackDocument
from .base_vector_store import BaseVectorStore, VectorStoreError, VectorStoreConnectionError


class FAISSVectorStore(BaseVectorStore):
    """
    基于FAISS的向量存储实现。
    
    提供高性能的向量相似度搜索，支持多种索引类型和持久化存储。
    适用于中到大规模的向量检索场景。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化FAISS向量存储。
        
        Args:
            config: FAISS配置字典
        """
        super().__init__(config)
        
        if not FAISS_AVAILABLE:
            raise VectorStoreConnectionError("FAISS未安装，请运行: pip install faiss-cpu")
        
        # FAISS特定配置
        faiss_config = config.get("faiss", {})
        self.storage_path = faiss_config.get("storage_path", "./data/faiss")
        self.index_type = faiss_config.get("index_type", "Flat")
        self.metric_type = faiss_config.get("metric_type", "INNER_PRODUCT")  # L2, INNER_PRODUCT
        self.auto_save_interval = faiss_config.get("auto_save_interval", 300)  # 5分钟
        self.use_gpu = faiss_config.get("use_gpu", False) and hasattr(faiss, 'StandardGpuResources')
        
        # 索引参数
        self.index_params = faiss_config.get("index_params", {})
        self.search_params = faiss_config.get("search_params", {})
        
        # 内部状态
        self.index = None
        self.documents = {}  # doc_id -> document mapping
        self.id_to_idx = {}  # doc_id -> faiss index mapping
        self.idx_to_id = {}  # faiss index -> doc_id mapping
        self.next_idx = 0
        
        # GPU资源（如果使用GPU）
        self.gpu_resources = None
        
        # 确保存储目录存在
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.logger.info(f"FAISSVectorStore初始化完成: {self.index_type}, 维度={self.vector_dim}")
    
    def initialize(self) -> bool:
        """
        初始化FAISS索引。
        
        Returns:
            初始化成功返回True，失败返回False
        """
        try:
            # 创建FAISS索引
            self.index = self._create_index()
            
            # 尝试加载已存在的索引
            index_path = os.path.join(self.storage_path, "index.faiss")
            if os.path.exists(index_path):
                self.logger.info("发现已存在的FAISS索引，正在加载...")
                self.load_index(self.storage_path)
            
            self.is_initialized = True
            self.logger.info("FAISS索引初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化FAISS索引失败: {str(e)}")
            return False
    
    def _create_index(self):
        """
        根据配置创建FAISS索引。
        
        Returns:
            FAISS索引对象
        """
        # 根据相似度函数选择度量类型
        if self.similarity_function == "cosine":
            metric = faiss.METRIC_INNER_PRODUCT  # 对于归一化向量，内积等同于余弦相似度
        elif self.similarity_function == "euclidean":
            metric = faiss.METRIC_L2
        else:  # dot_product
            metric = faiss.METRIC_INNER_PRODUCT
        
        # 创建基础索引
        if self.index_type.lower() == "flat":
            index = faiss.IndexFlatIP(self.vector_dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.vector_dim)
        
        elif self.index_type.startswith("IVF"):
            # 解析IVF参数 (例如: "IVF1024,Flat")
            parts = self.index_type.split(",")
            nlist = int(parts[0].replace("IVF", ""))
            quantizer_type = parts[1] if len(parts) > 1 else "Flat"
            
            # 创建量化器
            if quantizer_type.lower() == "flat":
                quantizer = faiss.IndexFlatIP(self.vector_dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.vector_dim)
            else:
                raise ValueError(f"不支持的量化器类型: {quantizer_type}")
            
            # 创建IVF索引
            index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist, metric)
            
        elif self.index_type.startswith("HNSW"):
            # HNSW参数
            M = self.index_params.get("M", 16)
            index = faiss.IndexHNSWFlat(self.vector_dim, M, metric)
            if "efConstruction" in self.index_params:
                index.hnsw.efConstruction = self.index_params["efConstruction"]
            
        else:
            self.logger.warning(f"未知索引类型 {self.index_type}，使用Flat索引")
            index = faiss.IndexFlatIP(self.vector_dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.vector_dim)
        
        # GPU支持
        if self.use_gpu:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
                self.logger.info("FAISS GPU模式已启用")
            except Exception as e:
                self.logger.warning(f"GPU初始化失败，回退到CPU模式: {str(e)}")
        
        return index
    
    def add_documents(
        self, 
        documents: List[HaystackDocument],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """
        批量添加文档到FAISS索引。
        
        Args:
            documents: Haystack文档列表
            embeddings: 对应的向量嵌入列表
            
        Returns:
            添加成功返回True，失败返回False
        """
        if not self.is_initialized:
            self.logger.error("FAISS索引未初始化")
            return False
        
        if not documents:
            return True
        
        if embeddings and len(documents) != len(embeddings):
            self.logger.error("文档数量与嵌入向量数量不匹配")
            return False
        
        try:
            vectors_to_add = []
            doc_ids = []
            
            for i, doc in enumerate(documents):
                # 获取或生成文档ID
                doc_id = doc.id if doc.id else f"doc_{self.next_idx}"
                
                # 检查向量嵌入
                if embeddings:
                    embedding = embeddings[i]
                elif doc.embedding:
                    embedding = doc.embedding
                else:
                    self.logger.warning(f"文档 {doc_id} 缺少向量嵌入，跳过")
                    continue
                
                # 验证向量
                if not self.validate_embedding(embedding):
                    self.logger.warning(f"文档 {doc_id} 的向量嵌入无效，跳过")
                    continue
                
                # 余弦相似度需要归一化向量
                if self.similarity_function == "cosine":
                    embedding = self._normalize_vector(embedding)
                
                vectors_to_add.append(embedding)
                doc_ids.append(doc_id)
                
                # 存储文档和映射关系
                self.documents[doc_id] = doc
                self.id_to_idx[doc_id] = self.next_idx
                self.idx_to_id[self.next_idx] = doc_id
                self.next_idx += 1
            
            # 添加向量到索引
            if vectors_to_add:
                vectors_array = np.array(vectors_to_add, dtype=np.float32)
                
                # 如果是IVF索引且未训练，需要先训练
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    if len(vectors_to_add) >= 256:  # FAISS建议最少256个向量用于训练
                        self.index.train(vectors_array)
                        self.logger.info("FAISS索引训练完成")
                    else:
                        self.logger.warning("向量数量不足，无法训练IVF索引")
                        return False
                
                self.index.add(vectors_array)
                self.logger.info(f"成功添加 {len(vectors_to_add)} 个文档到FAISS索引")
                
                # 自动保存
                self._auto_save()
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"添加文档到FAISS索引失败: {str(e)}")
            return False
    
    def search(
        self, 
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[HaystackDocument, float]]:
        """
        在FAISS索引中进行向量搜索。
        
        Args:
            query_embedding: 查询向量
            top_k: 返回的最相似文档数量
            filters: 元数据过滤条件（FAISS本身不支持，需要后处理）
            
        Returns:
            包含(文档, 相似度得分)的元组列表
        """
        if not self.is_initialized:
            self.logger.error("FAISS索引未初始化")
            return []
        
        if not self.validate_embedding(query_embedding):
            self.logger.error("查询向量无效")
            return []
        
        try:
            # 准备查询向量
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # 余弦相似度需要归一化
            if self.similarity_function == "cosine":
                query_vector = self._normalize_vector(query_vector.flatten()).reshape(1, -1)
            
            # 设置搜索参数
            if hasattr(self.index, 'nprobe') and "nprobe" in self.search_params:
                self.index.nprobe = self.search_params["nprobe"]
            
            if hasattr(self.index, 'hnsw') and "ef" in self.search_params:
                self.index.hnsw.efSearch = self.search_params["ef"]
            
            # 执行搜索
            scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示无效结果
                    break
                
                doc_id = self.idx_to_id.get(idx)
                if doc_id and doc_id in self.documents:
                    document = self.documents[doc_id]
                    
                    # 应用元数据过滤（如果提供）
                    if filters and not self._apply_filters(document, filters):
                        continue
                    
                    # 转换分数（FAISS内积分数可能需要调整）
                    similarity_score = float(score)
                    if self.similarity_function == "euclidean":
                        # L2距离转换为相似度
                        similarity_score = 1.0 / (1.0 + similarity_score)
                    
                    results.append((document, similarity_score))
            
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"FAISS搜索失败: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[HaystackDocument]:
        """
        根据文档ID获取文档。
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档对象，不存在则返回None
        """
        return self.documents.get(doc_id)
    
    def update_document(
        self, 
        doc_id: str,
        document: HaystackDocument,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        更新文档（FAISS不支持原地更新，需要重建索引）。
        
        Args:
            doc_id: 文档ID
            document: 新的文档对象
            embedding: 新的向量嵌入
            
        Returns:
            更新成功返回True，失败返回False
        """
        # FAISS不支持单个向量的原地更新
        # 这里采用简单的删除+添加策略
        # 对于频繁更新的场景，建议使用支持更新的存储后端
        
        if doc_id not in self.documents:
            self.logger.warning(f"文档 {doc_id} 不存在，无法更新")
            return False
        
        try:
            # 删除旧文档（仅从映射中删除，FAISS索引项保留）
            del self.documents[doc_id]
            
            # 添加新文档
            document.id = doc_id
            return self.add_documents([document], [embedding] if embedding else None)
            
        except Exception as e:
            self.logger.error(f"更新文档 {doc_id} 失败: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档（FAISS不支持删除，仅从映射中移除）。
        
        Args:
            doc_id: 文档ID
            
        Returns:
            删除成功返回True，失败返回False
        """
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                
                # 清理映射关系
                if doc_id in self.id_to_idx:
                    idx = self.id_to_idx[doc_id]
                    del self.id_to_idx[doc_id]
                    if idx in self.idx_to_id:
                        del self.idx_to_id[idx]
                
                self.logger.info(f"文档 {doc_id} 已从映射中移除")
                return True
            else:
                self.logger.warning(f"文档 {doc_id} 不存在")
                return False
                
        except Exception as e:
            self.logger.error(f"删除文档 {doc_id} 失败: {str(e)}")
            return False
    
    def delete_all_documents(self) -> bool:
        """
        删除所有文档。
        
        Returns:
            删除成功返回True，失败返回False
        """
        try:
            # 重置索引
            self.index = self._create_index()
            
            # 清空所有映射
            self.documents.clear()
            self.id_to_idx.clear()
            self.idx_to_id.clear()
            self.next_idx = 0
            
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
        return len(self.documents)
    
    def save_index(self, path: str) -> bool:
        """
        保存FAISS索引和元数据到文件。
        
        Args:
            path: 保存目录路径
            
        Returns:
            保存成功返回True，失败返回False
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # 保存FAISS索引
            index_path = os.path.join(path, "index.faiss")
            if self.use_gpu:
                # GPU索引需要先移到CPU
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
            
            # 保存文档映射
            metadata = {
                "documents": {doc_id: {
                    "content": doc.content,
                    "meta": doc.meta,
                    "embedding": doc.embedding
                } for doc_id, doc in self.documents.items()},
                "id_to_idx": self.id_to_idx,
                "idx_to_id": {str(k): v for k, v in self.idx_to_id.items()},
                "next_idx": self.next_idx,
                "config": self.config
            }
            
            metadata_path = os.path.join(path, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"FAISS索引已保存到 {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存FAISS索引失败: {str(e)}")
            return False
    
    def load_index(self, path: str) -> bool:
        """
        从文件加载FAISS索引和元数据。
        
        Args:
            path: 索引文件目录路径
            
        Returns:
            加载成功返回True，失败返回False
        """
        try:
            index_path = os.path.join(path, "index.faiss")
            metadata_path = os.path.join(path, "metadata.json")
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                self.logger.warning(f"索引文件不完整: {path}")
                return False
            
            # 加载FAISS索引
            index = faiss.read_index(index_path)
            
            # GPU支持
            if self.use_gpu:
                try:
                    if not self.gpu_resources:
                        self.gpu_resources = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
                except Exception as e:
                    self.logger.warning(f"GPU加载失败，使用CPU模式: {str(e)}")
            
            self.index = index
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 重构文档对象
            self.documents = {}
            for doc_id, doc_data in metadata["documents"].items():
                doc = HaystackDocument(
                    content=doc_data["content"],
                    meta=doc_data["meta"],
                    embedding=doc_data.get("embedding")
                )
                doc.id = doc_id
                self.documents[doc_id] = doc
            
            self.id_to_idx = metadata["id_to_idx"]
            self.idx_to_id = {int(k): v for k, v in metadata["idx_to_id"].items()}
            self.next_idx = metadata["next_idx"]
            
            self.is_initialized = True
            self.logger.info(f"FAISS索引已从 {path} 加载，包含 {len(self.documents)} 个文档")
            return True
            
        except Exception as e:
            self.logger.error(f"加载FAISS索引失败: {str(e)}")
            return False
    
    def _normalize_vector(self, vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        归一化向量（用于余弦相似度）。
        
        Args:
            vector: 输入向量
            
        Returns:
            归一化后的向量
        """
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _apply_filters(self, document: HaystackDocument, filters: Dict[str, Any]) -> bool:
        """
        应用元数据过滤条件。
        
        Args:
            document: 文档对象
            filters: 过滤条件
            
        Returns:
            满足条件返回True，否则返回False
        """
        try:
            for key, value in filters.items():
                if key in document.meta:
                    doc_value = document.meta[key]
                    if isinstance(value, list):
                        if doc_value not in value:
                            return False
                    elif doc_value != value:
                        return False
                else:
                    return False
            return True
        except Exception:
            return False
    
    def _auto_save(self):
        """
        自动保存索引（如果启用）。
        """
        if self.auto_save_interval > 0:
            try:
                self.save_index(self.storage_path)
            except Exception as e:
                self.logger.warning(f"自动保存失败: {str(e)}")
    
    def close(self):
        """
        关闭FAISS存储，释放资源。
        """
        try:
            # 保存索引
            if self.is_initialized:
                self.save_index(self.storage_path)
            
            # 释放GPU资源
            if self.gpu_resources:
                del self.gpu_resources
                self.gpu_resources = None
            
            super().close()
            
        except Exception as e:
            self.logger.error(f"关闭FAISS存储失败: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息。
        
        Returns:
            包含索引统计信息的字典
        """
        if not self.is_initialized:
            return {}
        
        stats = {
            "total_vectors": self.index.ntotal,
            "vector_dimension": self.vector_dim,
            "index_type": self.index_type,
            "is_trained": getattr(self.index, 'is_trained', True),
            "use_gpu": self.use_gpu
        }
        
        return stats