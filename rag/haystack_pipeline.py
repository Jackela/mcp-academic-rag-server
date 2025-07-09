"""
Haystack Pipeline模块 - 实现基于Haystack的RAG管道
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, Callable
import json

from haystack import Pipeline
from haystack.components.generators import OpenAIChatGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores import DocumentStore

from .prompt_builder import ChatPromptBuilder
from ..connectors.haystack_llm_connector import HaystackLLMConnector

# 配置日志
logger = logging.getLogger(__name__)

class RAGPipeline:
    """基于Haystack的RAG管道"""
    
    def __init__(
        self,
        llm_connector: HaystackLLMConnector,
        document_store: Optional[DocumentStore] = None,
        retriever_top_k: int = 5,
        prompt_builder = None
    ):
        """
        初始化RAG管道
        
        Args:
            llm_connector (HaystackLLMConnector): LLM连接器
            document_store (DocumentStore, optional): 文档存储，如果为None则创建InMemoryDocumentStore
            retriever_top_k (int): 检索器返回的最大文档数量
            prompt_builder (ChatPromptBuilder, optional): 提示构建器，如果为None则创建默认构建器
        """
        self.llm_connector = llm_connector
        self.document_store = document_store or InMemoryDocumentStore()
        self.retriever_top_k = retriever_top_k
        self.prompt_builder = prompt_builder or ChatPromptBuilder()
        
        # 创建检索器
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store, top_k=retriever_top_k)
        
        # 创建Pipeline
        self._create_pipeline()
    
    def _create_pipeline(self):
        """创建Haystack Pipeline"""
        self.pipeline = Pipeline()
        
        # 向Pipeline中添加组件
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm_connector.generator)
        
        # 定义组件之间的连接
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")
        
        logger.info(f"成功创建Haystack RAG Pipeline")
    
    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        运行RAG管道
        
        Args:
            query (str): 用户查询
            chat_history (list, optional): 聊天历史记录，格式为[{"role": "user", "content": "消息内容"}]
            filters (dict, optional): 检索过滤条件
            generation_kwargs (dict, optional): 生成参数
            
        Returns:
            dict: 包含回答和检索文档的结果
        """
        try:
            # 准备输入
            inputs = {
                "retriever": {"query": query, "filters": filters},
                "prompt_builder": {
                    "query": query,
                    "chat_history": chat_history or []
                }
            }
            
            # 运行管道
            logger.info(f"运行RAG管道: 查询='{query}'")
            output = self.pipeline.run(inputs)
            
            # 提取结果
            answer = output["llm"]["replies"][0].content
            documents = output["retriever"]["documents"]
            
            # 构建结果
            result = {
                "answer": answer,
                "documents": [{
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                } for doc in documents],
                "query": query
            }
            
            logger.info(f"RAG管道运行成功: 检索到{len(documents)}个文档")
            return result
            
        except Exception as e:
            logger.error(f"RAG管道运行失败: {str(e)}")
            return {
                "answer": f"查询处理失败: {str(e)}",
                "documents": [],
                "query": query,
                "error": str(e)
            }
    
    def update_retriever(self, document_store: DocumentStore = None, top_k: int = None):
        """
        更新检索器
        
        Args:
            document_store (DocumentStore, optional): 新的文档存储
            top_k (int, optional): 新的最大检索文档数量
        """
        # 更新文档存储
        if document_store:
            self.document_store = document_store
        
        # 更新top_k
        if top_k:
            self.retriever_top_k = top_k
        
        # 重新创建检索器
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store, top_k=self.retriever_top_k)
        
        # 重新创建Pipeline
        self._create_pipeline()
        
        logger.info(f"已更新检索器: top_k={self.retriever_top_k}")
    
    def update_llm_connector(self, llm_connector: HaystackLLMConnector):
        """
        更新LLM连接器
        
        Args:
            llm_connector (HaystackLLMConnector): 新的LLM连接器
        """
        self.llm_connector = llm_connector
        self._create_pipeline()
        logger.info(f"已更新LLM连接器: model={llm_connector.model}")


class RAGPipelineFactory:
    """RAG管道工厂类，创建基于Haystack的RAG管道"""
    
    @staticmethod
    def create_pipeline(
        llm_connector: HaystackLLMConnector,
        document_store: Optional[DocumentStore] = None,
        retriever_top_k: int = 5,
        prompt_builder = None,
        config: Optional[Dict[str, Any]] = None
    ) -> RAGPipeline:
        """
        创建RAG管道
        
        Args:
            llm_connector (HaystackLLMConnector): LLM连接器
            document_store (DocumentStore, optional): 文档存储
            retriever_top_k (int): 检索器返回的最大文档数量
            prompt_builder (ChatPromptBuilder, optional): 提示构建器
            config (dict, optional): 配置参数
            
        Returns:
            RAGPipeline: RAG管道实例
        """
        # 处理配置参数
        if config:
            retriever_top_k = config.get("retriever_top_k", retriever_top_k)
        
        return RAGPipeline(
            llm_connector=llm_connector,
            document_store=document_store,
            retriever_top_k=retriever_top_k,
            prompt_builder=prompt_builder
        )
