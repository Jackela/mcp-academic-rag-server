"""
基于Haystack框架的检索器实现。

该模块提供了HaystackRetriever类，基于InMemoryEmbeddingRetriever实现
高效的文档检索功能，支持语义搜索和相似度排序。
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import time

from haystack.nodes import SentenceTransformersDocumentEmbedder, EmbeddingRetriever
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
        
        # 初始化检索器
        self._initialize_retriever()
        
        self.logger.info(f"HaystackRetriever初始化完成，使用模型: {self.model_name_or_path}")
    
    def _initialize_retriever(self) -> None:
        """
        初始化Haystack检索器。
        """
        try:
            # 初始化检索器
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=self.model_name_or_path,
                top_k=self.top_k,
                model_format="sentence_transformers",
                batch_size=self.batch_size
            )
            
            self.logger.info("成功初始化Haystack检索器")
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
            
            # 执行检索
            retriever_results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            # 转换结果格式
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
            
            # 执行批量检索
            batch_results = self.retriever.retrieve_batch(
                queries=queries,
                top_k=top_k,
                filters=filters
            )
            
            # 转换结果格式
            formatted_results = []
            
            for query_results in batch_results:
                query_formatted = []
                
                for doc in query_results:
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
                    
                    query_formatted.append(result)
                
                formatted_results.append(query_formatted)
            
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
            
            # 更新其他配置
            if "top_k" in config:
                self.top_k = config["top_k"]
            
            if "threshold" in config:
                self.threshold = config["threshold"]
            
            # 更新配置字典
            self.config.update(config)
            
            # 如果需要，重新初始化检索器
            if need_reinit:
                self._initialize_retriever()
            
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
            "document_count": self.document_store.get_document_count()
        }


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
