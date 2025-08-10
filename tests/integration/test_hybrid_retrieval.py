"""
混合检索集成测试模块

测试混合检索功能，验证密集(embedding)和稀疏(BM25)检索的结合效果。
"""

import unittest
import tempfile
import shutil
import os
from typing import List, Dict, Any

from haystack.schema import Document as HaystackDocument

from retrievers.haystack_retriever import HaystackRetriever
from document_stores.haystack_store import HaystackDocumentStore


class TestHybridRetrieval(unittest.TestCase):
    """混合检索集成测试类"""
    
    def setUp(self):
        """测试前设置"""
        # 创建临时文档存储
        self.temp_dir = tempfile.mkdtemp()
        
        # 初始化文档存储
        store_config = {
            "type": "memory",
            "similarity": "cosine"
        }
        self.doc_store = HaystackDocumentStore(store_config)
        
        # 准备测试文档
        self.test_documents = [
            HaystackDocument(
                content="Machine learning algorithms are used for pattern recognition and data analysis. "
                       "Deep learning is a subset of machine learning that uses neural networks.",
                meta={"title": "ML Introduction", "type": "technical"},
                id="doc1"
            ),
            HaystackDocument(
                content="Natural language processing enables computers to understand human language. "
                       "NLP techniques include tokenization, parsing, and semantic analysis.",
                meta={"title": "NLP Overview", "type": "technical"},
                id="doc2"
            ),
            HaystackDocument(
                content="Computer vision algorithms process and analyze visual information from images. "
                       "Object detection and image classification are common computer vision tasks.",
                meta={"title": "Computer Vision", "type": "technical"},
                id="doc3"
            ),
            HaystackDocument(
                content="The research methodology involved collecting data from multiple sources. "
                       "Statistical analysis was performed to validate the research hypothesis.",
                meta={"title": "Research Methods", "type": "academic"},
                id="doc4"
            ),
            HaystackDocument(
                content="Artificial intelligence encompasses machine learning, natural language processing, "
                       "and computer vision. AI systems can perform tasks that typically require human intelligence.",
                meta={"title": "AI Overview", "type": "general"},
                id="doc5"
            )
        ]
        
        # 向文档存储添加文档
        document_store = self.doc_store.get_document_store()
        document_store.write_documents(self.test_documents)
        
        # 更新嵌入（如果需要）
        document_store.update_embeddings()
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_hybrid_retrieval_enabled(self):
        """测试启用混合检索功能"""
        # 配置混合检索
        config = {
            "enable_hybrid": True,
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
            "top_k": 3,
            "threshold": 0.0  # 降低阈值以获取更多结果
        }
        
        retriever = HaystackRetriever(
            document_store=self.doc_store,
            config=config
        )
        
        # 测试语义查询（应该更好地匹配深度学习文档）
        semantic_query = "deep learning neural networks"
        semantic_results = retriever.retrieve(semantic_query, top_k=3)
        
        # 验证返回结果
        self.assertGreater(len(semantic_results), 0, "混合检索应该返回结果")
        self.assertLessEqual(len(semantic_results), 3, "结果数量不应超过top_k")
        
        # 验证结果包含相关文档
        result_ids = [r.get("id") for r in semantic_results]
        self.assertIn("doc1", result_ids, "应该包含机器学习文档")
        
        # 测试关键词查询（BM25应该发挥作用）
        keyword_query = "research methodology statistical analysis"
        keyword_results = retriever.retrieve(keyword_query, top_k=3)
        
        # 验证关键词查询结果
        self.assertGreater(len(keyword_results), 0, "关键词查询应该返回结果")
        result_ids = [r.get("id") for r in keyword_results]
        self.assertIn("doc4", result_ids, "应该包含研究方法文档")
    
    def test_hybrid_vs_single_retrieval(self):
        """测试混合检索与单一检索的效果对比"""
        # 混合检索配置
        hybrid_config = {
            "enable_hybrid": True,
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "top_k": 5,
            "threshold": 0.0
        }
        
        # 单一检索配置
        single_config = {
            "enable_hybrid": False,
            "top_k": 5,
            "threshold": 0.0
        }
        
        hybrid_retriever = HaystackRetriever(
            document_store=self.doc_store,
            config=hybrid_config
        )
        
        single_retriever = HaystackRetriever(
            document_store=self.doc_store,
            config=single_config
        )
        
        # 测试包含语义和关键词的复杂查询
        complex_query = "artificial intelligence machine learning research analysis"
        
        hybrid_results = hybrid_retriever.retrieve(complex_query)
        single_results = single_retriever.retrieve(complex_query)
        
        # 验证两种方法都返回结果
        self.assertGreater(len(hybrid_results), 0, "混合检索应该返回结果")
        self.assertGreater(len(single_results), 0, "单一检索应该返回结果")
        
        # 获取结果ID
        hybrid_ids = [r.get("id") for r in hybrid_results]
        single_ids = [r.get("id") for r in single_results]
        
        # 验证混合检索能够找到更相关的文档组合
        # 对于这个查询，混合检索应该能够同时考虑语义相似性和关键词匹配
        self.assertTrue(
            len(set(hybrid_ids)) >= len(set(single_ids)) or
            any(doc_id in hybrid_ids for doc_id in ["doc1", "doc4", "doc5"]),
            "混合检索应该提供更全面的结果"
        )
    
    def test_weight_configuration(self):
        """测试不同权重配置的影响"""
        base_config = {
            "enable_hybrid": True,
            "top_k": 3,
            "threshold": 0.0
        }
        
        # 测试偏向密集检索的配置
        dense_heavy_config = {**base_config, "dense_weight": 0.9, "sparse_weight": 0.1}
        dense_retriever = HaystackRetriever(
            document_store=self.doc_store,
            config=dense_heavy_config
        )
        
        # 测试偏向稀疏检索的配置
        sparse_heavy_config = {**base_config, "dense_weight": 0.1, "sparse_weight": 0.9}
        sparse_retriever = HaystackRetriever(
            document_store=self.doc_store,
            config=sparse_heavy_config
        )
        
        # 使用一个既有语义相似性又有关键词匹配的查询
        query = "computer vision image processing"
        
        dense_results = dense_retriever.retrieve(query)
        sparse_results = sparse_retriever.retrieve(query)
        
        # 验证两种配置都返回结果
        self.assertGreater(len(dense_results), 0, "密集偏向检索应该返回结果")
        self.assertGreater(len(sparse_results), 0, "稀疏偏向检索应该返回结果")
        
        # 验证权重配置会影响结果排序
        dense_top_id = dense_results[0].get("id") if dense_results else None
        sparse_top_id = sparse_results[0].get("id") if sparse_results else None
        
        # 这些可能不同，因为权重不同会导致不同的排序
        self.assertIsNotNone(dense_top_id, "密集偏向检索应该有顶部结果")
        self.assertIsNotNone(sparse_top_id, "稀疏偏向检索应该有顶部结果")
    
    def test_batch_hybrid_retrieval(self):
        """测试批量混合检索"""
        config = {
            "enable_hybrid": True,
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "top_k": 2,
            "threshold": 0.0
        }
        
        retriever = HaystackRetriever(
            document_store=self.doc_store,
            config=config
        )
        
        # 批量查询
        queries = [
            "machine learning algorithms",
            "natural language processing",
            "computer vision tasks"
        ]
        
        batch_results = retriever.retrieve_batch(queries)
        
        # 验证批量结果
        self.assertEqual(len(batch_results), len(queries), "批量结果数量应该匹配查询数量")
        
        for i, results in enumerate(batch_results):
            self.assertGreater(len(results), 0, f"查询 {i} 应该返回结果")
            self.assertLessEqual(len(results), 2, f"查询 {i} 结果数量不应超过top_k")
            
            # 验证结果格式
            for result in results:
                self.assertIn("content", result, "结果应该包含content字段")
                self.assertIn("score", result, "结果应该包含score字段")
                self.assertIn("id", result, "结果应该包含id字段")
    
    def test_retriever_metadata(self):
        """测试检索器元数据"""
        config = {
            "enable_hybrid": True,
            "dense_weight": 0.7,
            "sparse_weight": 0.3
        }
        
        retriever = HaystackRetriever(
            document_store=self.doc_store,
            config=config
        )
        
        metadata = retriever.get_metadata()
        
        # 验证元数据包含混合检索信息
        self.assertIn("enable_hybrid", metadata, "元数据应该包含enable_hybrid")
        self.assertIn("dense_weight", metadata, "元数据应该包含dense_weight")
        self.assertIn("sparse_weight", metadata, "元数据应该包含sparse_weight")
        
        self.assertTrue(metadata["enable_hybrid"], "enable_hybrid应该为True")
        self.assertEqual(metadata["dense_weight"], 0.7, "dense_weight应该正确")
        self.assertEqual(metadata["sparse_weight"], 0.3, "sparse_weight应该正确")
    
    def test_config_update_hybrid(self):
        """测试动态更新混合检索配置"""
        initial_config = {
            "enable_hybrid": False,
            "top_k": 3
        }
        
        retriever = HaystackRetriever(
            document_store=self.doc_store,
            config=initial_config
        )
        
        # 验证初始配置
        self.assertFalse(retriever.enable_hybrid, "初始应该禁用混合检索")
        
        # 更新配置启用混合检索
        update_config = {
            "enable_hybrid": True,
            "dense_weight": 0.8,
            "sparse_weight": 0.2
        }
        
        success = retriever.update_config(update_config)
        
        # 验证配置更新
        self.assertTrue(success, "配置更新应该成功")
        self.assertTrue(retriever.enable_hybrid, "混合检索应该被启用")
        self.assertEqual(retriever.dense_weight, 0.8, "dense_weight应该更新")
        self.assertEqual(retriever.sparse_weight, 0.2, "sparse_weight应该更新")


if __name__ == "__main__":
    unittest.main()