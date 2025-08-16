"""
Processing Pipeline Module - Document Processing Workflow Management

Provides the Pipeline class for combining multiple processors and executing
document processing workflows in sequence.
"""

import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import IProcessor
from utils.performance_enhancements import MemoryManager, profile_performance


class Pipeline:
    """
    Processing Pipeline for document workflow management.

    This class coordinates processors in a specific sequence for document processing.
    It provides methods for adding/removing processors, reordering them, and processing documents.

    This is a core system component implementing the Pipeline Pattern, connecting
    a series of processing steps to form a complete processing workflow.
    """

    def __init__(self, name: str = "DefaultPipeline"):
        """
        Initialize Pipeline object.

        Args:
            name: Pipeline name, defaults to "DefaultPipeline"
        """
        self.name = name
        self.processors: List[IProcessor] = []
        self._memory_manager = MemoryManager()
        self._executor = ThreadPoolExecutor(max_workers=4)

    def add_processor(self, processor: IProcessor) -> None:
        """
        Add a processor to the pipeline.

        Args:
            processor: IProcessor object to add
        """
        self.processors.append(processor)
        logger.info("Processor added to pipeline", processor_name=processor.get_name(), pipeline_name=self.name)

    def remove_processor(self, processor_name: str) -> bool:
        """
        从流水线中移除指定名称的处理器。

        Args:
            processor_name: 要移除的处理器名称

        Returns:
            如果成功移除处理器则返回True，否则返回False
        """
        initial_count = len(self.processors)
        self.processors = [p for p in self.processors if p.get_name() != processor_name]
        removed = len(self.processors) < initial_count

        if removed:
            logger.info("Processor removed from pipeline", processor_name=processor_name, pipeline_name=self.name)
        else:
            logger.warning("Processor not found in pipeline", processor_name=processor_name, pipeline_name=self.name)

        return removed

    def get_processors(self) -> List[IProcessor]:
        """
        获取流水线中的所有处理器。

        Returns:
            处理器列表
        """
        return self.processors.copy()

    def clear_processors(self) -> None:
        """
        清空流水线中的所有处理器。
        """
        self.processors.clear()
        logger.info(f"流水线 '{self.name}' 中的所有处理器已清空")

    def reorder_processors(self, processor_names: List[str]) -> bool:
        """
        根据提供的名称列表重新排序处理器。

        Args:
            processor_names: 处理器名称列表，按期望的顺序排列

        Returns:
            如果成功重排序则返回True，否则返回False
        """
        # 创建名称到处理器的映射
        processor_map = {p.get_name(): p for p in self.processors}

        # 检查所有名称是否都存在
        if not all(name in processor_map for name in processor_names):
            logger.error("重排序失败：某些处理器名称不存在")
            return False

        # 按提供的顺序重排处理器
        try:
            self.processors = [processor_map[name] for name in processor_names]
            logger.info(f"流水线 '{self.name}' 中的处理器已重新排序")
            return True
        except Exception as e:
            logger.error(f"重排序失败：{str(e)}")
            return False

    async def process_document(self, document: Document, start_from: Optional[str] = None) -> ProcessResult:
        """
        异步处理文档，按顺序执行流水线中的处理器。

        Args:
            document: 要处理的Document对象
            start_from: 起始处理器名称，如果指定，则从该处理器开始处理，默认为None

        Returns:
            表示处理结果的ProcessResult对象
        """
        if not self.processors:
            return ProcessResult.error_result("流水线中没有处理器")

        document.update_status("processing")
        logger.info(f"开始异步处理文档: {document.document_id} - {document.file_name}")

        start_processing = False if start_from else True

        for processor in self.processors:
            processor_name = processor.get_name()

            # 如果指定了起始处理器，则跳过之前的处理器
            if not start_processing:
                if processor_name == start_from:
                    start_processing = True
                else:
                    continue

            if not processor.supports_file_type(document.file_type):
                logger.warning(f"处理器 '{processor_name}' 不支持文件类型 '{document.file_type}'，已跳过")
                continue

            logger.info(f"使用处理器 '{processor_name}' 异步处理文档 {document.document_id}")

            try:
                # 检查处理器是否支持异步处理
                if hasattr(processor, "process_async") and callable(processor.process_async):
                    # 使用异步方法
                    result = await processor.process_async(document)
                else:
                    # 使用专用线程池执行器避免阻塞，限制并发线程数
                    if not hasattr(self, "_executor"):
                        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline-worker")

                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self._executor, processor.process, document)

                if not result.is_successful():
                    document.update_status("error")
                    error_msg = f"处理器 '{processor_name}' 处理失败: {result.get_message()}"
                    logger.error(error_msg)
                    return ProcessResult.error_result(error_msg, result.get_error())

                # 记录处理历史
                document.processing_history.append(
                    {
                        "processor": processor_name,
                        "stage": processor.get_stage(),
                        "success": True,
                        "message": result.get_message(),
                    }
                )

                logger.info(f"处理器 '{processor_name}' 成功异步处理文档 {document.document_id}")

            except Exception as e:
                document.update_status("error")
                error_msg = f"处理器 '{processor_name}' 发生异常: {str(e)}"
                logger.exception(error_msg)
                return ProcessResult.error_result(error_msg, e)

        document.update_status("completed")
        logger.info(f"文档 {document.document_id} 异步处理完成")
        return ProcessResult.success_result("文档处理完成")

    async def process_documents(self, documents: List[Document], max_concurrent: int = 5) -> Dict[str, ProcessResult]:
        """
        异步批量处理多个文档，支持并发处理以提高性能。

        Args:
            documents: 要处理的Document对象列表
            max_concurrent: 最大并发处理数量，默认为5

        Returns:
            字典，键为文档ID，值为对应的ProcessResult对象
        """
        logger.info(f"开始异步批量处理 {len(documents)} 个文档，最大并发数: {max_concurrent}")

        # 创建信号量来限制并发数量
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(document: Document) -> tuple[str, ProcessResult]:
            """带信号量控制的文档处理"""
            async with semaphore:
                result = await self.process_document(document)
                return document.document_id, result

        # 创建所有任务
        tasks = [process_with_semaphore(doc) for doc in documents]

        # 并发执行所有任务
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        # 整理结果
        results = {}
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                # 处理异常情况
                logger.error(f"文档处理任务异常: {str(task_result)}")
                # 为异常创建错误结果（需要文档ID，这里使用通用ID）
                results[f"error_{len(results)}"] = ProcessResult.error_result(
                    f"任务执行异常: {str(task_result)}", task_result
                )
            else:
                doc_id, result = task_result
                results[doc_id] = result

        logger.info(f"异步批量处理完成，成功处理 {len([r for r in results.values() if r.is_successful()])} 个文档")
        return results

    def process_document_sync(self, document: Document, start_from: Optional[str] = None) -> ProcessResult:
        """
        同步处理文档（保持向后兼容性）。

        Args:
            document: 要处理的Document对象
            start_from: 起始处理器名称，如果指定，则从该处理器开始处理，默认为None

        Returns:
            表示处理结果的ProcessResult对象
        """
        if not self.processors:
            return ProcessResult.error_result("流水线中没有处理器")

        document.update_status("processing")
        logger.info(f"开始同步处理文档: {document.document_id} - {document.file_name}")

        start_processing = False if start_from else True

        for processor in self.processors:
            processor_name = processor.get_name()

            # 如果指定了起始处理器，则跳过之前的处理器
            if not start_processing:
                if processor_name == start_from:
                    start_processing = True
                else:
                    continue

            if not processor.supports_file_type(document.file_type):
                logger.warning(f"处理器 '{processor_name}' 不支持文件类型 '{document.file_type}'，已跳过")
                continue

            logger.info(f"使用处理器 '{processor_name}' 同步处理文档 {document.document_id}")

            try:
                result = processor.process(document)

                if not result.is_successful():
                    document.update_status("error")
                    error_msg = f"处理器 '{processor_name}' 处理失败: {result.get_message()}"
                    logger.error(error_msg)
                    return ProcessResult.error_result(error_msg, result.get_error())

                # 记录处理历史
                document.processing_history.append(
                    {
                        "processor": processor_name,
                        "stage": processor.get_stage(),
                        "success": True,
                        "message": result.get_message(),
                    }
                )

                logger.info(f"处理器 '{processor_name}' 成功同步处理文档 {document.document_id}")

            except Exception as e:
                document.update_status("error")
                error_msg = f"处理器 '{processor_name}' 发生异常: {str(e)}"
                logger.exception(error_msg)
                return ProcessResult.error_result(error_msg, e)

        document.update_status("completed")
        logger.info(f"文档 {document.document_id} 同步处理完成")
        return ProcessResult.success_result("文档处理完成")

    def process_documents_sync(self, documents: List[Document]) -> Dict[str, ProcessResult]:
        """
        同步批量处理多个文档（保持向后兼容性）。

        Args:
            documents: 要处理的Document对象列表

        Returns:
            字典，键为文档ID，值为对应的ProcessResult对象
        """
        results = {}

        for document in documents:
            results[document.document_id] = self.process_document_sync(document)

        return results
