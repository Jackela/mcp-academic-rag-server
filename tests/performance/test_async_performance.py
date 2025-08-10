"""
异步性能测试模块

比较同步和异步文档处理管道的性能差异，验证异步架构的性能提升。
"""

import asyncio
import time
import unittest
import tempfile
import os
import shutil
from typing import List
from unittest.mock import MagicMock, patch

# 添加项目根目录到系统路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.pipeline import Pipeline
from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor


class MockProcessor(BaseProcessor):
    """模拟处理器，用于性能测试"""
    
    def __init__(self, name: str, processing_time: float = 0.1):
        """
        初始化模拟处理器
        
        Args:
            name: 处理器名称
            processing_time: 模拟处理时间（秒）
        """
        super().__init__(name=name, description=f"Mock processor {name}")
        self.processing_time = processing_time
    
    def process(self, document: Document) -> ProcessResult:
        """同步处理文档"""
        # 模拟处理时间
        time.sleep(self.processing_time)
        
        # 模拟处理结果
        result_data = {
            "processor": self.get_name(),
            "document_id": document.document_id,
            "processing_time": self.processing_time
        }
        
        document.store_content(self.get_stage(), result_data)
        return ProcessResult.success_result(f"Mock processing by {self.get_name()}", result_data)
    
    async def process_async(self, document: Document) -> ProcessResult:
        """异步处理文档"""
        # 使用异步sleep模拟处理时间
        await asyncio.sleep(self.processing_time)
        
        # 模拟处理结果
        result_data = {
            "processor": self.get_name(),
            "document_id": document.document_id,
            "processing_time": self.processing_time,
            "async": True
        }
        
        document.store_content(self.get_stage(), result_data)
        return ProcessResult.success_result(f"Async mock processing by {self.get_name()}", result_data)


class IOBoundMockProcessor(BaseProcessor):
    """模拟IO密集型处理器"""
    
    def __init__(self, name: str, io_time: float = 0.2):
        super().__init__(name=name, description=f"IO-bound mock processor {name}")
        self.io_time = io_time
    
    def process(self, document: Document) -> ProcessResult:
        """同步IO处理"""
        # 模拟IO操作（文件读写等）
        time.sleep(self.io_time)
        
        result_data = {
            "processor": self.get_name(),
            "document_id": document.document_id,
            "io_time": self.io_time,
            "type": "io_bound"
        }
        
        document.store_content(self.get_stage(), result_data)
        return ProcessResult.success_result(f"IO processing by {self.get_name()}", result_data)
    
    async def process_async(self, document: Document) -> ProcessResult:
        """异步IO处理"""
        # 使用异步sleep模拟异步IO操作
        await asyncio.sleep(self.io_time)
        
        result_data = {
            "processor": self.get_name(),
            "document_id": document.document_id,
            "io_time": self.io_time,
            "type": "io_bound",
            "async": True
        }
        
        document.store_content(self.get_stage(), result_data)
        return ProcessResult.success_result(f"Async IO processing by {self.get_name()}", result_data)


class TestAsyncPerformance(unittest.TestCase):
    """异步性能测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试文档
        self.test_documents = []
        for i in range(10):
            doc = Document(f"test_doc_{i}.txt")
            doc.file_type = "text"
            doc.file_path = os.path.join(self.temp_dir, f"test_doc_{i}.txt")
            
            # 创建实际文件
            with open(doc.file_path, 'w', encoding='utf-8') as f:
                f.write(f"Test document {i} content for performance testing.")
            
            self.test_documents.append(doc)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_single_document_sync_vs_async(self):
        """测试单文档处理的同步vs异步性能"""
        # 创建管道
        sync_pipeline = Pipeline("SyncPipeline")
        async_pipeline = Pipeline("AsyncPipeline")
        
        # 添加处理器
        processors = [
            MockProcessor("processor_1", 0.1),
            MockProcessor("processor_2", 0.1),
            MockProcessor("processor_3", 0.1)
        ]
        
        for processor in processors:
            sync_pipeline.add_processor(processor)
            async_pipeline.add_processor(processor)
        
        document = self.test_documents[0]
        
        # 测试同步处理
        start_time = time.time()
        sync_result = sync_pipeline.process_document_sync(document)
        sync_duration = time.time() - start_time
        
        # 重置文档状态
        document.content.clear()
        document.processing_history.clear()
        document.status = "created"
        
        # 测试异步处理
        async def run_async_test():
            start_time = time.time()
            async_result = await async_pipeline.process_document(document)
            return async_result, time.time() - start_time
        
        async_result, async_duration = asyncio.run(run_async_test())
        
        # 验证结果
        self.assertTrue(sync_result.is_successful())
        self.assertTrue(async_result.is_successful())
        
        # 对于单个文档，同步和异步处理时间应该相近
        # 但异步版本有轻微的开销
        print(f"单文档处理 - 同步耗时: {sync_duration:.3f}s, 异步耗时: {async_duration:.3f}s")
        
        # 验证处理结果一致性
        self.assertEqual(len(document.processing_history), 3)
    
    def test_batch_processing_performance_improvement(self):
        """测试批量处理的性能提升"""
        # 创建管道
        sync_pipeline = Pipeline("SyncBatchPipeline") 
        async_pipeline = Pipeline("AsyncBatchPipeline")
        
        # 添加模拟IO密集型处理器
        processors = [
            IOBoundMockProcessor("io_processor_1", 0.05),
            IOBoundMockProcessor("io_processor_2", 0.05)
        ]
        
        for processor in processors:
            sync_pipeline.add_processor(processor)
            async_pipeline.add_processor(processor)
        
        # 准备测试文档
        test_docs = self.test_documents[:5]  # 使用5个文档进行测试
        
        # 测试同步批量处理
        start_time = time.time()
        sync_results = sync_pipeline.process_documents_sync(test_docs)
        sync_duration = time.time() - start_time
        
        # 重置文档状态
        for doc in test_docs:
            doc.content.clear()
            doc.processing_history.clear()
            doc.status = "created"
        
        # 测试异步批量处理
        async def run_async_batch_test():
            start_time = time.time()
            async_results = await async_pipeline.process_documents(test_docs, max_concurrent=3)
            return async_results, time.time() - start_time
        
        async_results, async_duration = asyncio.run(run_async_batch_test())
        
        # 验证结果
        self.assertEqual(len(sync_results), len(test_docs))
        self.assertEqual(len(async_results), len(test_docs))
        
        for result in sync_results.values():
            self.assertTrue(result.is_successful())
        
        for result in async_results.values():
            self.assertTrue(result.is_successful())
        
        # 异步处理应该显著快于同步处理
        performance_improvement = (sync_duration - async_duration) / sync_duration * 100
        
        print(f"批量处理性能比较:")
        print(f"  同步处理耗时: {sync_duration:.3f}s")
        print(f"  异步处理耗时: {async_duration:.3f}s")
        print(f"  性能提升: {performance_improvement:.1f}%")
        
        # 异步处理应该至少快30%（由于并发处理）
        self.assertGreater(performance_improvement, 30,
                          f"异步处理应该显著快于同步处理，实际性能提升: {performance_improvement:.1f}%")
    
    def test_concurrent_processing_scalability(self):
        """测试并发处理的可扩展性"""
        async_pipeline = Pipeline("ConcurrencyTestPipeline")
        
        # 添加处理器
        processor = IOBoundMockProcessor("concurrent_processor", 0.1)
        async_pipeline.add_processor(processor)
        
        # 测试不同并发级别
        document_counts = [5, 10, 15]
        concurrency_levels = [1, 3, 5]
        
        results = {}
        
        for doc_count in document_counts:
            results[doc_count] = {}
            test_docs = self.test_documents[:doc_count]
            
            for concurrency in concurrency_levels:
                # 重置文档状态
                for doc in test_docs:
                    doc.content.clear()
                    doc.processing_history.clear()
                    doc.status = "created"
                
                # 测试异步处理
                async def run_concurrency_test():
                    start_time = time.time()
                    async_results = await async_pipeline.process_documents(
                        test_docs, max_concurrent=concurrency
                    )
                    return async_results, time.time() - start_time
                
                async_results, duration = asyncio.run(run_concurrency_test())
                results[doc_count][concurrency] = duration
                
                # 验证所有文档都被成功处理
                self.assertEqual(len(async_results), doc_count)
                for result in async_results.values():
                    self.assertTrue(result.is_successful())
        
        # 输出性能结果
        print("\\n并发处理性能测试结果:")
        print("文档数量 | 并发级别=1 | 并发级别=3 | 并发级别=5 | 性能提升(1->5)")
        print("-" * 60)
        
        for doc_count in document_counts:
            seq_time = results[doc_count][1]
            max_conc_time = results[doc_count][5]
            improvement = (seq_time - max_conc_time) / seq_time * 100
            
            print(f"{doc_count:8d} | {results[doc_count][1]:9.3f}s | "
                  f"{results[doc_count][3]:9.3f}s | {results[doc_count][5]:9.3f}s | "
                  f"{improvement:8.1f}%")
            
            # 验证并发处理确实提高了性能
            self.assertLess(max_conc_time, seq_time,
                           f"对于{doc_count}个文档，并发处理应该快于顺序处理")
    
    def test_memory_usage_async_vs_sync(self):
        """测试异步vs同步处理的内存使用情况"""
        import psutil
        import gc
        
        # 创建大量文档进行测试
        large_doc_set = []
        for i in range(20):
            doc = Document(f"large_test_doc_{i}.txt")
            doc.file_type = "text"
            doc.file_path = os.path.join(self.temp_dir, f"large_test_doc_{i}.txt")
            
            # 创建较大的测试内容
            large_content = "Test content " * 1000  # 约13KB的内容
            with open(doc.file_path, 'w', encoding='utf-8') as f:
                f.write(large_content)
            
            large_doc_set.append(doc)
        
        # 创建管道
        sync_pipeline = Pipeline("MemoryTestSyncPipeline")
        async_pipeline = Pipeline("MemoryTestAsyncPipeline")
        
        # 添加处理器
        processor = MockProcessor("memory_test_processor", 0.01)
        sync_pipeline.add_processor(processor)
        async_pipeline.add_processor(processor)
        
        # 测试同步处理的内存使用
        gc.collect()  # 清理垃圾回收
        process = psutil.Process()
        
        memory_before_sync = process.memory_info().rss / 1024 / 1024  # MB
        sync_results = sync_pipeline.process_documents_sync(large_doc_set)
        memory_after_sync = process.memory_info().rss / 1024 / 1024  # MB
        sync_memory_usage = memory_after_sync - memory_before_sync
        
        # 重置文档状态
        for doc in large_doc_set:
            doc.content.clear()
            doc.processing_history.clear()
            doc.status = "created"
        
        # 测试异步处理的内存使用
        gc.collect()  # 清理垃圾回收
        
        async def run_memory_test():
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            async_results = await async_pipeline.process_documents(large_doc_set, max_concurrent=5)
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            return async_results, memory_after - memory_before
        
        async_results, async_memory_usage = asyncio.run(run_memory_test())
        
        # 验证处理结果
        self.assertEqual(len(sync_results), len(large_doc_set))
        self.assertEqual(len(async_results), len(large_doc_set))
        
        print(f"\\n内存使用比较:")
        print(f"  同步处理内存增量: {sync_memory_usage:.2f} MB")
        print(f"  异步处理内存增量: {async_memory_usage:.2f} MB")
        
        # 异步处理可能使用稍多内存（由于并发），但不应该过多
        # 这主要是为了记录和观察，不做严格断言
        if async_memory_usage > sync_memory_usage * 1.5:
            print(f"警告: 异步处理使用了显著更多内存 ({async_memory_usage/sync_memory_usage:.1f}x)")
    
    def test_error_handling_in_async_pipeline(self):
        """测试异步管道中的错误处理"""
        
        class ErrorProneProcessor(BaseProcessor):
            """会出错的处理器"""
            
            def __init__(self, name: str, error_probability: float = 0.3):
                super().__init__(name=name)
                self.error_probability = error_probability
                self.processed_count = 0
            
            def process(self, document: Document) -> ProcessResult:
                self.processed_count += 1
                if self.processed_count % 3 == 0:  # 每3个文档出一次错
                    return ProcessResult.error_result("Simulated processing error")
                
                time.sleep(0.05)
                result_data = {"processor": self.get_name()}
                document.store_content(self.get_stage(), result_data)
                return ProcessResult.success_result("Success")
            
            async def process_async(self, document: Document) -> ProcessResult:
                self.processed_count += 1
                if self.processed_count % 3 == 0:  # 每3个文档出一次错
                    return ProcessResult.error_result("Simulated async processing error")
                
                await asyncio.sleep(0.05)
                result_data = {"processor": self.get_name(), "async": True}
                document.store_content(self.get_stage(), result_data)
                return ProcessResult.success_result("Async success")
        
        # 创建管道
        async_pipeline = Pipeline("ErrorHandlingPipeline")
        error_processor = ErrorProneProcessor("error_prone_processor")
        async_pipeline.add_processor(error_processor)
        
        # 测试错误处理
        test_docs = self.test_documents[:6]  # 使用6个文档，应该有2个失败
        
        async def run_error_test():
            results = await async_pipeline.process_documents(test_docs, max_concurrent=3)
            return results
        
        results = asyncio.run(run_error_test())
        
        # 验证结果
        self.assertEqual(len(results), len(test_docs))
        
        success_count = sum(1 for result in results.values() if result.is_successful())
        error_count = sum(1 for result in results.values() if not result.is_successful())
        
        print(f"\\n错误处理测试结果:")
        print(f"  成功处理: {success_count}/{len(test_docs)}")
        print(f"  处理失败: {error_count}/{len(test_docs)}")
        
        # 应该有一些成功，一些失败
        self.assertGreater(success_count, 0, "应该有一些文档成功处理")
        self.assertGreater(error_count, 0, "应该有一些文档处理失败")


class PerformanceBenchmark:
    """性能基准测试工具"""
    
    @staticmethod
    def run_comprehensive_benchmark():
        """运行综合性能基准测试"""
        print("=" * 80)
        print("异步文档处理管道性能基准测试")
        print("=" * 80)
        
        # 运行测试套件
        suite = unittest.TestSuite()
        suite.addTest(TestAsyncPerformance('test_single_document_sync_vs_async'))
        suite.addTest(TestAsyncPerformance('test_batch_processing_performance_improvement'))
        suite.addTest(TestAsyncPerformance('test_concurrent_processing_scalability'))
        suite.addTest(TestAsyncPerformance('test_memory_usage_async_vs_sync'))
        suite.addTest(TestAsyncPerformance('test_error_handling_in_async_pipeline'))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        print("\\n" + "=" * 80)
        print(f"基准测试完成 - 成功: {result.testsRun - len(result.failures) - len(result.errors)}, "
              f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
        print("=" * 80)
        
        return result


if __name__ == "__main__":
    # 运行单独的测试或基准测试
    import argparse
    
    parser = argparse.ArgumentParser(description="异步性能测试")
    parser.add_argument("--benchmark", action="store_true", help="运行性能基准测试")
    parser.add_argument("--test", type=str, help="运行特定测试")
    
    args = parser.parse_args()
    
    if args.benchmark:
        PerformanceBenchmark.run_comprehensive_benchmark()
    elif args.test:
        suite = unittest.TestSuite()
        suite.addTest(TestAsyncPerformance(args.test))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        # 运行所有测试
        unittest.main(verbosity=2)