"""
Milvus持久化存储集成测试

该测试验证Milvus持久化存储功能，包括文档存储、检索和系统重启后的数据持久性。
"""

import asyncio
import unittest
import tempfile
import os
import shutil
import time
import json
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from document_stores.milvus_store import MilvusDocumentStore, MILVUS_AVAILABLE
from document_stores.haystack_store import HaystackDocumentStore
from core.config_manager import ConfigManager
from models.document import Document
from haystack.schema import Document as HaystackDocument


class TestMilvusPersistence(unittest.TestCase):
    """Milvus持久化存储集成测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类级别的设置"""
        if not MILVUS_AVAILABLE:
            raise unittest.SkipTest("Milvus不可用，跳过持久化测试")
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Milvus测试配置
        self.milvus_config = {
            "host": "localhost",
            "port": 19530,
            "user": "",
            "password": "",
            "database": "default",
            "collection_name": f"test_persistence_{int(time.time())}",  # 使用时间戳避免冲突
            "vector_dimension": 384,
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "index_params": {"nlist": 128},
            "search_params": {"nprobe": 10}
        }
        
        # 文档存储配置
        self.document_store_config = {
            "type": "milvus",
            "embedding_dim": 384,
            "similarity": "cosine",
            "return_embedding": True,
            "milvus": self.milvus_config
        }
        
        # 创建测试文档
        self.test_documents = self._create_test_documents()
        
        # 第一个Milvus存储实例
        self.milvus_store = None
        self.haystack_store = None
    
    def tearDown(self):
        """测试后清理"""
        # 清理Milvus集合
        if self.milvus_store:
            try:
                self.milvus_store.close()
            except Exception:
                pass
        
        if self.haystack_store and self.haystack_store.get_milvus_store():
            try:
                self.haystack_store.get_milvus_store().close()
            except Exception:
                pass
        
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_documents(self) -> List[Document]:
        """创建测试文档"""
        documents = []
        
        # 创建多个测试文档
        test_contents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret and understand visual information.",
            "Reinforcement learning teaches agents to make decisions through trial and error."
        ]
        
        for i, content in enumerate(test_contents):
            doc = Document(f"test_doc_{i}.txt")
            doc.file_type = "text"
            doc.file_path = os.path.join(self.temp_dir, f"test_doc_{i}.txt")
            
            # 创建实际文件
            with open(doc.file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 添加处理后的内容
            doc.store_content("OCRProcessor", content)
            doc.add_metadata("topic", f"AI_Topic_{i}")
            doc.add_metadata("difficulty", "beginner" if i < 3 else "advanced")
            
            documents.append(doc)
        
        return documents
    
    def _create_haystack_documents_with_embeddings(self) -> List[HaystackDocument]:
        """创建带有嵌入向量的Haystack文档"""
        import numpy as np
        
        haystack_docs = []
        
        for i, doc in enumerate(self.test_documents):
            # 生成模拟嵌入向量
            embedding = np.random.rand(384).tolist()
            
            haystack_doc = HaystackDocument(
                content=doc.get_content("OCRProcessor"),
                meta={
                    "file_path": doc.file_path,
                    "file_name": doc.file_name,
                    "file_type": doc.file_type,
                    "original_id": doc.document_id,
                    "metadata": doc.metadata,
                    "tags": doc.tags
                },
                embedding=embedding
            )
            
            haystack_docs.append(haystack_doc)
        
        return haystack_docs
    
    def test_milvus_document_store_persistence(self):
        """测试MilvusDocumentStore的持久化功能"""
        # 第一阶段：创建存储并添加文档
        print("\\n=== 第一阶段：添加文档到Milvus ===")
        
        self.milvus_store = MilvusDocumentStore(self.milvus_config)
        haystack_docs = self._create_haystack_documents_with_embeddings()
        
        # 添加文档
        success = self.milvus_store.add_documents(haystack_docs)
        self.assertTrue(success, "应该成功添加文档到Milvus")
        
        # 验证文档数量
        doc_count = self.milvus_store.get_document_count()
        self.assertEqual(doc_count, len(haystack_docs), f"文档数量应该为{len(haystack_docs)}")
        
        # 测试搜索功能
        query_embedding = haystack_docs[0].embedding
        search_results = self.milvus_store.search(query_embedding, top_k=3)
        self.assertGreater(len(search_results), 0, "应该能够搜索到结果")
        
        # 记录第一个文档的ID用于后续验证
        first_doc_id = haystack_docs[0].id
        
        print(f"成功添加 {doc_count} 个文档到Milvus集合")
        print(f"搜索返回 {len(search_results)} 个结果")
        
        # 关闭第一个连接
        self.milvus_store.close()
        
        # 第二阶段：重新连接并验证数据持久性
        print("\\n=== 第二阶段：重新连接并验证持久性 ===")
        
        # 创建新的存储实例（模拟系统重启）
        new_milvus_store = MilvusDocumentStore(self.milvus_config)
        
        try:
            # 验证文档仍然存在
            persistent_doc_count = new_milvus_store.get_document_count()
            self.assertEqual(persistent_doc_count, len(haystack_docs), 
                           f"重启后文档数量应该仍为{len(haystack_docs)}")
            
            # 验证可以检索特定文档
            retrieved_doc = new_milvus_store.get_document_by_id(first_doc_id)
            self.assertIsNotNone(retrieved_doc, "应该能够检索到之前存储的文档")
            
            # 验证搜索功能仍然工作
            search_results_after_restart = new_milvus_store.search(query_embedding, top_k=3)
            self.assertGreater(len(search_results_after_restart), 0, "重启后应该仍能搜索到结果")
            
            print(f"重启后验证：文档数量 = {persistent_doc_count}")
            print(f"成功检索文档ID: {first_doc_id}")
            print(f"重启后搜索返回 {len(search_results_after_restart)} 个结果")
            
        finally:
            new_milvus_store.close()
    
    def test_haystack_document_store_milvus_integration(self):
        """测试HaystackDocumentStore与Milvus的集成"""
        print("\\n=== 测试HaystackDocumentStore Milvus集成 ===")
        
        # 创建HaystackDocumentStore，配置为使用Milvus
        config_manager = MagicMock()
        config_manager.get_value.return_value = self.document_store_config
        
        self.haystack_store = HaystackDocumentStore(config_manager=config_manager)
        
        # 验证使用了Milvus存储
        self.assertTrue(self.haystack_store.is_using_milvus(), "应该使用Milvus存储")
        
        # 添加文档
        success_count, fail_count = self.haystack_store.add_documents(self.test_documents)
        self.assertEqual(success_count, len(self.test_documents), "所有文档都应该成功添加")
        self.assertEqual(fail_count, 0, "不应该有失败的文档")
        
        # 验证文档数量
        doc_count = self.haystack_store.get_document_count()
        self.assertEqual(doc_count, len(self.test_documents), f"文档数量应该为{len(self.test_documents)}")
        
        print(f"通过HaystackDocumentStore成功添加 {success_count} 个文档")
        print(f"验证文档数量: {doc_count}")
    
    def test_milvus_error_handling(self):
        """测试Milvus错误处理"""
        print("\\n=== 测试Milvus错误处理 ===")
        
        # 测试无效配置
        invalid_config = self.milvus_config.copy()
        invalid_config["host"] = "invalid_host"
        invalid_config["port"] = 99999
        
        with self.assertRaises(Exception):
            MilvusDocumentStore(invalid_config)
        
        print("正确处理了无效连接配置")
    
    def test_milvus_data_integrity(self):
        """测试Milvus数据完整性"""
        print("\\n=== 测试Milvus数据完整性 ===")
        
        self.milvus_store = MilvusDocumentStore(self.milvus_config)
        haystack_docs = self._create_haystack_documents_with_embeddings()
        
        # 添加文档
        success = self.milvus_store.add_documents(haystack_docs)
        self.assertTrue(success)
        
        # 验证每个文档的数据完整性
        for original_doc in haystack_docs:
            retrieved_doc = self.milvus_store.get_document_by_id(original_doc.id)
            self.assertIsNotNone(retrieved_doc, f"应该能检索到文档 {original_doc.id}")
            
            # 验证内容
            self.assertEqual(retrieved_doc["content"], original_doc.content, "文档内容应该匹配")
            
            # 验证元数据结构
            self.assertIsInstance(retrieved_doc["metadata"], dict, "元数据应该是字典类型")
            
        print(f"验证了 {len(haystack_docs)} 个文档的数据完整性")
    
    def test_milvus_concurrent_access(self):
        """测试Milvus并发访问"""
        print("\\n=== 测试Milvus并发访问 ===")
        
        async def concurrent_test():
            # 创建多个存储实例
            stores = []
            for i in range(3):
                config = self.milvus_config.copy()
                config["collection_name"] = f"concurrent_test_{int(time.time())}_{i}"
                stores.append(MilvusDocumentStore(config))
            
            try:
                # 并发添加文档
                tasks = []
                for i, store in enumerate(stores):
                    docs = self._create_haystack_documents_with_embeddings()
                    # 修改文档ID以避免冲突
                    for doc in docs:
                        doc.id = f"{doc.id}_store_{i}"
                    
                    task = asyncio.create_task(
                        asyncio.to_thread(store.add_documents, docs)
                    )
                    tasks.append(task)
                
                # 等待所有任务完成
                results = await asyncio.gather(*tasks)
                
                # 验证所有操作都成功
                for result in results:
                    self.assertTrue(result, "并发添加文档应该成功")
                
                # 验证每个存储的文档数量
                for store in stores:
                    count = store.get_document_count()
                    self.assertEqual(count, len(self.test_documents), "每个存储应该有正确的文档数量")
                
                print(f"成功完成 {len(stores)} 个并发存储操作")
                
            finally:
                # 清理
                for store in stores:
                    try:
                        store.close()
                    except Exception:
                        pass
        
        # 运行异步测试
        asyncio.run(concurrent_test())
    
    def test_milvus_performance_benchmarks(self):
        """测试Milvus性能基准"""
        print("\\n=== Milvus性能基准测试 ===")
        
        self.milvus_store = MilvusDocumentStore(self.milvus_config)
        
        # 创建更多测试文档
        large_doc_set = []
        for i in range(50):  # 创建50个文档进行性能测试
            doc = self._create_haystack_documents_with_embeddings()[0]  # 使用第一个作为模板
            doc.id = f"perf_test_doc_{i}"
            doc.content = f"Performance test document {i} with content for benchmarking."
            large_doc_set.append(doc)
        
        # 测试批量插入性能
        start_time = time.time()
        success = self.milvus_store.add_documents(large_doc_set)
        insert_time = time.time() - start_time
        
        self.assertTrue(success, "批量插入应该成功")
        
        # 测试搜索性能
        query_embedding = large_doc_set[0].embedding
        
        start_time = time.time()
        search_results = self.milvus_store.search(query_embedding, top_k=10)
        search_time = time.time() - start_time
        
        self.assertGreater(len(search_results), 0, "搜索应该返回结果")
        
        # 输出性能指标
        print(f"性能指标:")
        print(f"  批量插入 {len(large_doc_set)} 个文档: {insert_time:.3f}s")
        print(f"  平均插入速度: {len(large_doc_set)/insert_time:.1f} docs/s")
        print(f"  搜索耗时: {search_time:.3f}s")
        print(f"  返回结果数量: {len(search_results)}")
        
        # 基本性能断言
        self.assertLess(insert_time, 30.0, "批量插入不应超过30秒")
        self.assertLess(search_time, 1.0, "搜索不应超过1秒")


class MilvusPersistenceTestSuite:
    """Milvus持久化测试套件"""
    
    @staticmethod
    def run_comprehensive_tests():
        """运行完整的持久化测试套件"""
        print("=" * 80)
        print("Milvus持久化存储集成测试套件")
        print("=" * 80)
        
        if not MILVUS_AVAILABLE:
            print("警告: Milvus不可用，跳过持久化测试")
            print("请确保已安装pymilvus并且Milvus服务正在运行")
            return
        
        # 创建测试套件
        suite = unittest.TestSuite()
        
        # 添加测试
        suite.addTest(TestMilvusPersistence('test_milvus_document_store_persistence'))
        suite.addTest(TestMilvusPersistence('test_haystack_document_store_milvus_integration'))
        suite.addTest(TestMilvusPersistence('test_milvus_error_handling'))
        suite.addTest(TestMilvusPersistence('test_milvus_data_integrity'))
        suite.addTest(TestMilvusPersistence('test_milvus_concurrent_access'))
        suite.addTest(TestMilvusPersistence('test_milvus_performance_benchmarks'))
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        print("\\n" + "=" * 80)
        print(f"持久化测试完成 - 成功: {result.testsRun - len(result.failures) - len(result.errors)}, "
              f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
        print("=" * 80)
        
        return result


if __name__ == "__main__":
    # 检查命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description="Milvus持久化测试")
    parser.add_argument("--suite", action="store_true", help="运行完整测试套件")
    parser.add_argument("--test", type=str, help="运行特定测试")
    
    args = parser.parse_args()
    
    if args.suite:
        MilvusPersistenceTestSuite.run_comprehensive_tests()
    elif args.test:
        suite = unittest.TestSuite()
        suite.addTest(TestMilvusPersistence(args.test))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        # 运行所有测试
        unittest.main(verbosity=2)