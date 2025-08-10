"""
基于Haystack框架的检索器实现。

该模块提供了HaystackRetriever类，基于InMemoryEmbeddingRetriever实现
高效的文档检索功能，支持语义搜索和相似度排序。
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import time

from haystack.nodes import SentenceTransformersDocumentEmbedder, EmbeddingRetriever, BM25Retriever
from haystack.schema import Document as HaystackDocument

from models.document import Document
from document_stores.haystack_store import HaystackDocumentStore


class HaystackRetriever:
    """
    基于Haystack框架的检索器类。
    
    该类封装了Haystack的EmbeddingRetriever功能，支持基于语义相似度的
    文档检索，并提供检索质量评估工具。
    
    通过配置可以调整检索参数、相似度阈值和排序策略等。
    """
    
    def __init__(
        self, 
        document_store: HaystackDocumentStore,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化HaystackRetriever对象。
        
        Args:
            document_store: Haystack文档存储对象
            model_name_or_path: 嵌入模型名称或路径，默认为"sentence-transformers/all-MiniLM-L6-v2"
            config: 检索器配置字典，默认为None
        """
        self.logger = logging.getLogger("haystack_retriever")
        self.config = config or {}
        
        # 获取文档存储
        self.document_store = document_store.get_document_store()
        
        # 从配置中获取参数
        self.model_name_or_path = self.config.get("model_name_or_path", model_name_or_path)
        self.top_k = self.config.get("top_k", 10)
        self.threshold = self.config.get("threshold", 0.5)
        self.batch_size = self.config.get("batch_size", 16)
        
        # 混合检索配置
        self.enable_hybrid = self.config.get("enable_hybrid", True)
        self.dense_weight = self.config.get("dense_weight", 0.7)
        self.sparse_weight = self.config.get("sparse_weight", 0.3)
        
        # 初始化检索器
        self._initialize_retrievers()
        
        self.logger.info(f"HaystackRetriever初始化完成，使用模型: {self.model_name_or_path}，混合检索: {self.enable_hybrid}")
    
    def _initialize_retrievers(self) -> None:
        """
        初始化Haystack检索器（包括密集和稀疏检索器）。
        """
        try:
            # 初始化嵌入检索器（密集检索）
            self.embedding_retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=self.model_name_or_path,
                top_k=self.top_k,
                model_format="sentence_transformers",
                batch_size=self.batch_size
            )
            
            # 初始化BM25检索器（稀疏检索）
            if self.enable_hybrid:
                self.bm25_retriever = BM25Retriever(
                    document_store=self.document_store,
                    top_k=self.top_k
                )
            
            # 保持向后兼容性
            self.retriever = self.embedding_retriever
            
            self.logger.info(f"成功初始化Haystack检索器，混合模式: {self.enable_hybrid}")
        except Exception as e:
            self.logger.error(f"初始化Haystack检索器失败: {str(e)}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        根据查询文本检索相关文档。
        
        Args:
            query: 查询文本
            top_k: 返回结果数量，默认使用初始化时设置的值
            filters: 过滤条件，默认为None
            
        Returns:
            检索结果列表，每个结果包含文档内容和元数据
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 使用参数或默认值
            if top_k is None:
                top_k = self.top_k
            
            if self.enable_hybrid:
                # 混合检索：结合密集和稀疏检索结果
                results = self._hybrid_retrieve(query, top_k, filters)
            else:
                # 单一检索：仅使用嵌入检索
                results = self._single_retrieve(query, top_k, filters)
            
            # 记录处理时间
            processing_time = time.time() - start_time
            
            self.logger.info(f"查询 '{query}' 检索完成，返回 {len(results)} 个结果，耗时 {processing_time:.2f} 秒")
            
            return results
            
        except Exception as e:
            self.logger.error(f"检索查询 '{query}' 时发生异常: {str(e)}")
            return []
    
    def retrieve_batch(self, queries: List[str], top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        """
        批量检索多个查询。
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回结果数量，默认使用初始化时设置的值
            filters: 过滤条件，默认为None
            
        Returns:
            检索结果列表的列表，每个内部列表对应一个查询的结果
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 使用参数或默认值
            if top_k is None:
                top_k = self.top_k
            
            # 批量处理每个查询
            formatted_results = []
            for query in queries:
                if self.enable_hybrid:
                    query_results = self._hybrid_retrieve(query, top_k, filters)
                else:
                    query_results = self._single_retrieve(query, top_k, filters)
                formatted_results.append(query_results)
            
            # 记录处理时间
            processing_time = time.time() - start_time
            
            self.logger.info(f"批量检索 {len(queries)} 个查询完成，耗时 {processing_time:.2f} 秒")
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"批量检索查询时发生异常: {str(e)}")
            return [[] for _ in range(len(queries))]
    
    def evaluate_retrieval(self, queries: List[str], relevant_docs: List[List[str]], top_k: Optional[int] = None) -> Dict[str, float]:
        """
        评估检索质量。
        
        Args:
            queries: 查询文本列表
            relevant_docs: 每个查询的相关文档ID列表的列表
            top_k: 评估使用的top_k值，默认使用初始化时设置的值
            
        Returns:
            包含评估指标的字典，如精确率、召回率和F1值
        """
        if top_k is None:
            top_k = self.top_k
        
        results = self.retrieve_batch(queries, top_k)
        
        # 计算评估指标
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0
        
        for i, (query_results, relevant) in enumerate(zip(results, relevant_docs)):
            # 提取检索结果ID
            retrieved_ids = []
            for result in query_results:
                doc_id = result.get("original_id") or result.get("id")
                if doc_id:
                    retrieved_ids.append(doc_id)
            
            # 计算相关性
            relevant_retrieved = set(retrieved_ids).intersection(set(relevant))
            num_relevant_retrieved = len(relevant_retrieved)
            
            # 计算精确率和召回率
            precision = num_relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
            recall = num_relevant_retrieved / len(relevant) if relevant else 0
            
            # 计算F1值
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
        
        # 计算平均值
        num_queries = len(queries)
        avg_precision = precision_sum / num_queries if num_queries > 0 else 0
        avg_recall = recall_sum / num_queries if num_queries > 0 else 0
        avg_f1 = f1_sum / num_queries if num_queries > 0 else 0
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新检索器配置。
        
        部分配置项更新后需要重新初始化检索器。
        
        Args:
            config: 新的配置字典
            
        Returns:
            如果成功更新则返回True，否则返回False
        """
        try:
            need_reinit = False
            
            # 检查关键配置是否变化
            if "model_name_or_path" in config and config["model_name_or_path"] != self.model_name_or_path:
                self.model_name_or_path = config["model_name_or_path"]
                need_reinit = True
            
            if "batch_size" in config and config["batch_size"] != self.batch_size:
                self.batch_size = config["batch_size"]
                need_reinit = True
                
            if "enable_hybrid" in config and config["enable_hybrid"] != self.enable_hybrid:
                self.enable_hybrid = config["enable_hybrid"]
                need_reinit = True
            
            # 更新其他配置
            if "top_k" in config:
                self.top_k = config["top_k"]
            
            if "threshold" in config:
                self.threshold = config["threshold"]
                
            if "dense_weight" in config:
                self.dense_weight = config["dense_weight"]
                
            if "sparse_weight" in config:
                self.sparse_weight = config["sparse_weight"]
            
            # 更新配置字典
            self.config.update(config)
            
            # 如果需要，重新初始化检索器
            if need_reinit:
                self._initialize_retrievers()
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新配置失败: {str(e)}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取检索器元数据，包括配置和状态信息。
        
        Returns:
            包含元数据的字典
        """
        return {
            "model_name": self.model_name_or_path,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "batch_size": self.batch_size,
            "enable_hybrid": self.enable_hybrid,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "document_count": self.document_store.get_document_count()
        }


    def _single_retrieve(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        单一检索模式（仅使用嵌入检索）。
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            检索结果列表
        """
        retriever_results = self.embedding_retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        return self._format_results(retriever_results)
    
    def _hybrid_retrieve(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        混合检索模式（结合密集和稀疏检索）。
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            检索结果列表
        """
        # 获取两倍的top_k结果，为混合提供更多选择
        retrieve_k = min(top_k * 2, 20)
        
        # 密集检索（语义相似度）
        dense_results = self.embedding_retriever.retrieve(
            query=query,
            top_k=retrieve_k,
            filters=filters
        )
        
        # 稀疏检索（关键词匹配）
        sparse_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=retrieve_k,
            filters=filters
        )
        
        # 合并和重新排序结果
        merged_results = self._merge_results(dense_results, sparse_results, top_k)
        
        return self._format_results(merged_results)
    
    def _merge_results(self, dense_results: List, sparse_results: List, top_k: int) -> List:
        """
        合并密集和稀疏检索结果。
        
        Args:
            dense_results: 密集检索结果
            sparse_results: 稀疏检索结果
            top_k: 最终返回结果数量
            
        Returns:
            合并后的结果列表
        """
        # 创建文档ID到结果的映射
        doc_scores = {}
        
        # 处理密集检索结果
        for doc in dense_results:
            doc_id = doc.id
            dense_score = doc.score if hasattr(doc, 'score') else 0.0
            doc_scores[doc_id] = {
                'doc': doc,
                'dense_score': dense_score,
                'sparse_score': 0.0,
                'combined_score': dense_score * self.dense_weight
            }
        
        # 处理稀疏检索结果
        for doc in sparse_results:
            doc_id = doc.id
            sparse_score = doc.score if hasattr(doc, 'score') else 0.0
            
            if doc_id in doc_scores:
                # 文档已存在，更新分数
                doc_scores[doc_id]['sparse_score'] = sparse_score
                doc_scores[doc_id]['combined_score'] = (
                    doc_scores[doc_id]['dense_score'] * self.dense_weight +
                    sparse_score * self.sparse_weight
                )
            else:
                # 新文档，添加到结果中
                doc_scores[doc_id] = {
                    'doc': doc,
                    'dense_score': 0.0,
                    'sparse_score': sparse_score,
                    'combined_score': sparse_score * self.sparse_weight
                }
        
        # 按综合分数排序
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        # 返回前top_k个结果，并更新分数
        final_results = []
        for item in sorted_results[:top_k]:
            doc = item['doc']
            # 更新文档的分数为综合分数
            doc.score = item['combined_score']
            final_results.append(doc)
        
        return final_results
    
    def _format_results(self, retriever_results: List) -> List[Dict[str, Any]]:
        """
        格式化检索结果。
        
        Args:
            retriever_results: 原始检索结果
            
        Returns:
            格式化后的结果列表
        """
        results = []
        for doc in retriever_results:
            # 只保留分数高于阈值的结果
            if hasattr(doc, 'score') and doc.score < self.threshold:
                continue
            
            result = {
                "content": doc.content,
                "score": doc.score if hasattr(doc, 'score') else None,
                "id": doc.id
            }
            
            # 添加元数据
            if hasattr(doc, 'meta'):
                result["meta"] = doc.meta
                
                # 如果存在原始ID，则添加到顶层
                if "original_id" in doc.meta:
                    result["original_id"] = doc.meta["original_id"]
            
            results.append(result)
        
        return results


# 工厂函数，便于创建检索器实例
def create_retriever(
    document_store: HaystackDocumentStore,
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    config: Optional[Dict[str, Any]] = None
) -> HaystackRetriever:
    """
    创建检索器实例的工厂函数。
    
    Args:
        document_store: Haystack文档存储对象
        model_name_or_path: 嵌入模型名称或路径，默认为"sentence-transformers/all-MiniLM-L6-v2"
        config: 检索器配置字典，默认为None
        
    Returns:
        HaystackRetriever实例
    """
    return HaystackRetriever(document_store, model_name_or_path, config)
