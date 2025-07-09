"""
基于Haystack框架的文档嵌入处理器实现。

该模块提供了HaystackEmbeddingProcessor类，使用SentenceTransformersDocumentEmbedder
生成文档的向量表示，并实现文档分块和批量处理功能。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import time

from haystack.nodes import SentenceTransformersDocumentEmbedder, PreProcessor
from haystack.schema import Document as HaystackDocument

from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor
from document_stores.haystack_store import HaystackDocumentStore
from utils.vector_utils import chunk_text


class HaystackEmbeddingProcessor(BaseProcessor):
    """
    基于Haystack框架的文档嵌入处理器类。
    
    该类使用SentenceTransformersDocumentEmbedder生成文档的向量表示，
    实现了文档分块和批量处理功能，以优化嵌入生成性能和质量。
    
    通过配置可以调整文档切分参数、嵌入模型和批处理大小等。
    """
    
    def __init__(
        self, 
        document_store: Optional[HaystackDocumentStore] = None,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化HaystackEmbeddingProcessor对象。
        
        Args:
            document_store: Haystack文档存储对象，默认为None
            model_name_or_path: 嵌入模型名称或路径，默认为"sentence-transformers/all-MiniLM-L6-v2"
            config: 处理器配置字典，默认为None
        """
        super().__init__(name="HaystackEmbeddingProcessor", description="使用Haystack生成文档嵌入向量表示", config=config or {})
        
        self.logger = logging.getLogger("haystack_embedding_processor")
        self.document_store = document_store
        
        # 从配置中获取参数
        self.model_name_or_path = self.config.get("model_name_or_path", model_name_or_path)
        self.chunk_size = self.config.get("chunk_size", 500)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.batch_size = self.config.get("batch_size", 32)
        self.progress_bar = self.config.get("progress_bar", True)
        
        # 初始化预处理器和嵌入生成器
        self._initialize_embedder()
        
        self.logger.info(f"HaystackEmbeddingProcessor初始化完成，使用模型: {self.model_name_or_path}")
    
    def _initialize_embedder(self) -> None:
        """
        初始化Haystack预处理器和嵌入生成器。
        """
        try:
            # 初始化文本预处理器
            self.preprocessor = PreProcessor(
                clean_empty_lines=True,
                clean_whitespace=True,
                clean_header_footer=True,
                split_by="word",
                split_length=self.chunk_size,
                split_overlap=self.chunk_overlap,
                split_respect_sentence_boundary=True
            )
            
            # 初始化嵌入生成器
            self.embedder = SentenceTransformersDocumentEmbedder(
                model_name_or_path=self.model_name_or_path,
                batch_size=self.batch_size,
                progress_bar=self.progress_bar
            )
            
            self.logger.info("成功初始化Haystack预处理器和嵌入生成器")
        except Exception as e:
            self.logger.error(f"初始化Haystack组件失败: {str(e)}")
            raise
    
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档，生成其向量表示。
        
        步骤：
        1. 从文档中提取文本内容
        2. 对文本进行预处理和分块
        3. 为每个文本块生成嵌入向量
        4. 将嵌入向量添加到文档存储（如果提供了文档存储）
        5. 将处理结果存储到文档的content字典中
        
        Args:
            document: 要处理的Document对象
            
        Returns:
            表示处理结果的ProcessResult对象
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 从文档中获取文本内容
            text = document.get_content("OCRProcessor") or document.get_content("StructureProcessor")
            
            if not text:
                return ProcessResult.error_result(f"文档 {document.document_id} 没有可用的文本内容")
            
            # 准备Haystack文档对象
            haystack_doc = HaystackDocument(
                content=text,
                meta={
                    "file_path": document.file_path,
                    "file_name": document.file_name,
                    "original_id": document.document_id
                }
            )
            
            # 预处理和分块
            chunked_docs = self.preprocessor.process([haystack_doc])
            
            if not chunked_docs:
                return ProcessResult.error_result(f"文档 {document.document_id} 预处理后没有生成任何块")
            
            self.logger.info(f"文档 {document.document_id} 被分成了 {len(chunked_docs)} 个块")
            
            # 生成嵌入向量
            docs_with_embeddings = self.embedder.embed(chunked_docs)
            
            # 提取嵌入向量和处理后的文本
            embeddings = [doc.embedding for doc in docs_with_embeddings]
            processed_texts = [doc.content for doc in docs_with_embeddings]
            
            # 计算平均嵌入向量（如需要）
            avg_embedding = None
            if embeddings:
                import numpy as np
                avg_embedding = np.mean(embeddings, axis=0).tolist()
            
            # 将结果存储到文档
            result_data = {
                "chunks": len(docs_with_embeddings),
                "processed_texts": processed_texts,
                "embeddings": embeddings,
                "average_embedding": avg_embedding
            }
            
            # 将处理后的文本块存储到文档中
            document.store_content(self.get_stage(), processed_texts)
            
            # 如果提供了文档存储，则添加文档及其嵌入向量
            if self.document_store:
                success = self.document_store.add_document(document, avg_embedding)
                if not success:
                    self.logger.warning(f"文档 {document.document_id} 未能添加到文档存储")
            
            # 记录处理时间
            processing_time = time.time() - start_time
            
            self.logger.info(f"文档 {document.document_id} 嵌入处理完成，耗时 {processing_time:.2f} 秒")
            
            return ProcessResult.success_result(
                f"文档嵌入生成成功：{len(docs_with_embeddings)} 个块",
                result_data
            )
            
        except Exception as e:
            self.logger.error(f"处理文档 {document.document_id} 时发生异常: {str(e)}")
            return ProcessResult.error_result(f"文档嵌入生成失败: {str(e)}", error=e)
    
    def batch_process(self, documents: List[Document]) -> Dict[str, ProcessResult]:
        """
        批量处理多个文档，优化嵌入生成性能。
        
        Args:
            documents: 要处理的Document对象列表
            
        Returns:
            字典，键为文档ID，值为对应的ProcessResult对象
        """
        results = {}
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 提取所有文档的文本内容并创建Haystack文档对象
            haystack_docs = []
            haystack_to_original = {}  # 映射Haystack文档到原始文档
            
            for doc in documents:
                text = doc.get_content("OCRProcessor") or doc.get_content("StructureProcessor")
                
                if not text:
                    results[doc.document_id] = ProcessResult.error_result(
                        f"文档 {doc.document_id} 没有可用的文本内容"
                    )
                    continue
                
                haystack_doc = HaystackDocument(
                    content=text,
                    meta={
                        "file_path": doc.file_path,
                        "file_name": doc.file_name,
                        "original_id": doc.document_id
                    }
                )
                
                haystack_docs.append(haystack_doc)
                haystack_to_original[haystack_doc.id] = doc
            
            if not haystack_docs:
                self.logger.warning("批处理中没有可处理的文档")
                return results
            
            # 预处理和分块
            chunked_docs = self.preprocessor.process(haystack_docs)
            
            if not chunked_docs:
                for doc in documents:
                    results[doc.document_id] = ProcessResult.error_result(
                        f"文档预处理后没有生成任何块"
                    )
                return results
            
            # 创建块到原始文档的映射
            chunk_to_original = {}
            for chunk in chunked_docs:
                original_id = chunk.meta.get("original_id")
                if original_id:
                    for doc in documents:
                        if doc.document_id == original_id:
                            chunk_to_original[chunk.id] = doc
                            break
            
            # 生成嵌入向量
            docs_with_embeddings = self.embedder.embed(chunked_docs)
            
            # 按原始文档分组结果
            doc_chunks = {}
            doc_embeddings = {}
            
            for doc in docs_with_embeddings:
                original_doc = chunk_to_original.get(doc.id)
                if not original_doc:
                    continue
                
                doc_id = original_doc.document_id
                
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                    doc_embeddings[doc_id] = []
                
                doc_chunks[doc_id].append(doc.content)
                doc_embeddings[doc_id].append(doc.embedding)
            
            # 处理每个文档的结果
            for doc in documents:
                doc_id = doc.document_id
                
                if doc_id not in doc_chunks:
                    if doc_id not in results:  # 可能已经在前面设置了错误结果
                        results[doc_id] = ProcessResult.error_result(
                            f"文档 {doc_id} 处理后没有生成块或嵌入向量"
                        )
                    continue
                
                # 计算平均嵌入向量
                import numpy as np
                avg_embedding = np.mean(doc_embeddings[doc_id], axis=0).tolist()
                
                # 存储处理结果
                doc.store_content(self.get_stage(), doc_chunks[doc_id])
                
                # 添加到文档存储
                if self.document_store:
                    self.document_store.add_document(doc, avg_embedding)
                
                # 创建成功结果
                result_data = {
                    "chunks": len(doc_chunks[doc_id]),
                    "processed_texts": doc_chunks[doc_id],
                    "embeddings": doc_embeddings[doc_id],
                    "average_embedding": avg_embedding
                }
                
                results[doc_id] = ProcessResult.success_result(
                    f"文档嵌入生成成功：{len(doc_chunks[doc_id])} 个块",
                    result_data
                )
            
            # 记录处理时间
            processing_time = time.time() - start_time
            
            self.logger.info(f"批量处理 {len(documents)} 个文档完成，耗时 {processing_time:.2f} 秒")
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量处理文档时发生异常: {str(e)}")
            
            # 为所有未处理的文档设置错误结果
            for doc in documents:
                if doc.document_id not in results:
                    results[doc.document_id] = ProcessResult.error_result(
                        f"文档嵌入生成失败: {str(e)}",
                        error=e
                    )
            
            return results
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新处理器配置。
        
        部分配置项更新后需要重新初始化嵌入生成器。
        
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
            
            if "chunk_size" in config and config["chunk_size"] != self.chunk_size:
                self.chunk_size = config["chunk_size"]
                need_reinit = True
            
            if "chunk_overlap" in config and config["chunk_overlap"] != self.chunk_overlap:
                self.chunk_overlap = config["chunk_overlap"]
                need_reinit = True
            
            if "batch_size" in config and config["batch_size"] != self.batch_size:
                self.batch_size = config["batch_size"]
                need_reinit = True
            
            # 更新配置
            self.config.update(config)
            
            # 如果需要，重新初始化组件
            if need_reinit:
                self._initialize_embedder()
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新配置失败: {str(e)}")
            return False
    
    def set_document_store(self, document_store: HaystackDocumentStore) -> None:
        """
        设置文档存储对象。
        
        Args:
            document_store: Haystack文档存储对象
        """
        self.document_store = document_store
        self.logger.info("已设置文档存储")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取处理器元数据，包括配置和状态信息。
        
        Returns:
            包含元数据的字典
        """
        return {
            "model_name": self.model_name_or_path,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "batch_size": self.batch_size,
            "document_store_connected": self.document_store is not None
        }


# 工厂函数，便于创建嵌入处理器实例
def create_embedding_processor(
    document_store: Optional[HaystackDocumentStore] = None,
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    config: Optional[Dict[str, Any]] = None
) -> HaystackEmbeddingProcessor:
    """
    创建嵌入处理器实例的工厂函数。
    
    Args:
        document_store: Haystack文档存储对象，默认为None
        model_name_or_path: 嵌入模型名称或路径，默认为"sentence-transformers/all-MiniLM-L6-v2"
        config: 处理器配置字典，默认为None
        
    Returns:
        HaystackEmbeddingProcessor实例
    """
    return HaystackEmbeddingProcessor(document_store, model_name_or_path, config)
