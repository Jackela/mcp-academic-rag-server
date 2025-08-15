"""
向量存储性能基准测试

详细的性能测试和基准比较，包括不同存储后端的性能特征、
扩展性测试和优化建议。
"""

import pytest
import time
import statistics
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import tempfile
import shutil

from haystack import Document as HaystackDocument

from document_stores.vector_store_factory import create_vector_store
from document_stores.implementations.base_vector_store import BaseVectorStore


class BenchmarkResult:
    """基准测试结果类"""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        self.timestamps = []
    
    def add_metric(self, key: str, value: float, unit: str = ""):
        """添加性能指标"""
        self.metrics[key] = {"value": value, "unit": unit}
    
    def add_timing(self, operation: str, duration: float):
        """添加时间测量"""
        if operation not in self.metrics:
            self.metrics[operation] = {"times": [], "unit": "seconds"}
        self.metrics[operation]["times"].append(duration)
    
    def calculate_stats(self):
        """计算统计数据"""
        for key, data in self.metrics.items():
            if "times" in data and data["times"]:
                times = data["times"]
                data.update({
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "std": statistics.stdev(times) if len(times) > 1 else 0,
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }


class VectorStorageBenchmark:
    """向量存储基准测试器"""
    
    def __init__(self):
        self.results = []
        self.temp_dirs = []
    
    def cleanup(self):
        """清理临时目录"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_temp_dir(self) -> str:
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def generate_documents(self, count: int, vector_dim: int, content_size: str = "medium") -> Tuple[List[HaystackDocument], List[List[float]]]:
        """生成测试文档"""
        content_templates = {
            "small": "Doc {i}: Small content.",
            "medium": "Doc {i}: This is a medium-sized document with some meaningful content for testing purposes. " * 2,
            "large": "Doc {i}: This is a large document with extensive content for performance testing. " * 10
        }
        
        template = content_templates.get(content_size, content_templates["medium"])
        docs = []
        embeddings = []
        
        for i in range(count):
            # 创建文档
            doc = HaystackDocument(
                content=template.format(i=i),
                meta={
                    "doc_id": i,
                    "category": f"category_{i % 10}",
                    "author": f"author_{i % 5}",
                    "size": content_size,
                    "batch": i // 100
                },
                id=f"benchmark_doc_{i}"
            )
            
            # 生成确定性向量嵌入
            np.random.seed(i % 1000)  # 限制种子范围以增加相似性
            embedding = np.random.random(vector_dim).tolist()
            
            docs.append(doc)
            embeddings.append(embedding)
        
        return docs, embeddings
    
    def benchmark_store_operations(self, store_config: Dict[str, Any], test_params: Dict[str, Any]) -> BenchmarkResult:
        """基准测试存储操作"""
        result = BenchmarkResult(f"{store_config['type']}_benchmark")
        
        # 创建存储
        if store_config["type"] == "faiss":
            store_config["faiss"]["storage_path"] = self.create_temp_dir()
        
        store = create_vector_store(store_config, auto_fallback=True)
        
        try:
            # 生成测试数据
            docs, embeddings = self.generate_documents(
                test_params["document_count"],
                test_params["vector_dimension"], 
                test_params.get("content_size", "medium")
            )
            
            # 测试初始化
            init_start = time.time()
            store.initialize()
            init_time = time.time() - init_start
            result.add_metric("initialization_time", init_time, "seconds")
            
            # 测试批量添加
            batch_size = test_params.get("batch_size", 100)
            add_times = []
            
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                start_time = time.time()
                success = store.add_documents(batch_docs, batch_embeddings)
                add_time = time.time() - start_time
                
                if success:
                    add_times.append(add_time)
                    result.add_timing("batch_add", add_time)
            
            total_add_time = sum(add_times)
            result.add_metric("total_add_time", total_add_time, "seconds")
            result.add_metric("docs_per_second", len(docs) / total_add_time if total_add_time > 0 else 0, "docs/sec")
            
            # 测试搜索性能
            search_times = []
            search_counts = test_params.get("search_iterations", 20)
            top_k = test_params.get("top_k", 10)
            
            for i in range(search_counts):
                # 使用不同的查询向量
                query_idx = i % len(embeddings)
                query_embedding = embeddings[query_idx]
                
                start_time = time.time()
                results = store.search(query_embedding, top_k=top_k)
                search_time = time.time() - start_time
                
                search_times.append(search_time)
                result.add_timing("search", search_time)
                
                # 验证结果质量
                if results and len(results) > 0:
                    top_score = results[0][1]
                    result.add_timing("top_score", top_score)
            
            avg_search_time = statistics.mean(search_times) if search_times else 0
            result.add_metric("avg_search_time", avg_search_time, "seconds")
            result.add_metric("searches_per_second", 1.0 / avg_search_time if avg_search_time > 0 else 0, "searches/sec")
            
            # 测试单个文档操作
            if len(docs) > 0:
                test_doc_id = docs[0].id
                
                # 获取文档
                start_time = time.time()
                retrieved_doc = store.get_document_by_id(test_doc_id)
                get_time = time.time() - start_time
                result.add_metric("get_document_time", get_time, "seconds")
                
                # 更新文档
                if retrieved_doc:
                    updated_doc = HaystackDocument(
                        content="Updated content for benchmark",
                        meta=retrieved_doc.meta,
                        id=test_doc_id
                    )
                    
                    start_time = time.time()
                    update_success = store.update_document(test_doc_id, updated_doc, embeddings[0])
                    update_time = time.time() - start_time
                    result.add_metric("update_document_time", update_time, "seconds")
                
                # 删除文档
                start_time = time.time()
                delete_success = store.delete_document(test_doc_id)
                delete_time = time.time() - start_time
                result.add_metric("delete_document_time", delete_time, "seconds")
            
            # 测试持久化（如果支持）
            if hasattr(store, 'save_index') and store_config.get("test_persistence", True):
                save_path = self.create_temp_dir()
                
                start_time = time.time()
                save_success = store.save_index(save_path)
                save_time = time.time() - start_time
                
                if save_success:
                    result.add_metric("save_index_time", save_time, "seconds")
                    
                    # 测试加载
                    start_time = time.time()
                    load_success = store.load_index(save_path)
                    load_time = time.time() - start_time
                    
                    if load_success:
                        result.add_metric("load_index_time", load_time, "seconds")
            
            # 添加存储信息
            storage_info = store.get_storage_info()
            result.add_metric("final_document_count", storage_info.get("document_count", 0), "documents")
            
        except Exception as e:
            result.add_metric("error", str(e), "error")
            
        finally:
            store.close()
        
        result.calculate_stats()
        return result
    
    def run_scalability_test(self, store_configs: List[Dict[str, Any]], document_counts: List[int]) -> List[BenchmarkResult]:
        """运行可扩展性测试"""
        results = []
        
        for store_config in store_configs:
            for doc_count in document_counts:
                test_params = {
                    "document_count": doc_count,
                    "vector_dimension": 384,
                    "batch_size": min(100, doc_count // 10 + 1),
                    "search_iterations": min(10, doc_count // 100 + 1),
                    "top_k": 5
                }
                
                print(f"Testing {store_config['type']} with {doc_count} documents...")
                result = self.benchmark_store_operations(store_config, test_params)
                result.name = f"{store_config['type']}_{doc_count}_docs"
                results.append(result)
        
        return results
    
    def run_dimension_impact_test(self, store_configs: List[Dict[str, Any]], dimensions: List[int]) -> List[BenchmarkResult]:
        """运行向量维度影响测试"""
        results = []
        
        for store_config in store_configs:
            for dim in dimensions:
                test_params = {
                    "document_count": 500,  # 固定文档数量
                    "vector_dimension": dim,
                    "batch_size": 50,
                    "search_iterations": 10,
                    "top_k": 5
                }
                
                print(f"Testing {store_config['type']} with {dim}D vectors...")
                result = self.benchmark_store_operations(store_config, test_params)
                result.name = f"{store_config['type']}_{dim}D"
                results.append(result)
        
        return results
    
    def run_concurrent_access_test(self, store_config: Dict[str, Any], concurrent_operations: int = 5) -> BenchmarkResult:
        """运行并发访问测试"""
        # 注意：这是一个简化的并发测试
        # 实际并发测试需要使用threading或multiprocessing
        
        result = BenchmarkResult(f"{store_config['type']}_concurrent")
        
        if store_config["type"] == "faiss":
            store_config["faiss"]["storage_path"] = self.create_temp_dir()
        
        store = create_vector_store(store_config, auto_fallback=True)
        
        try:
            store.initialize()
            
            # 准备测试数据
            docs, embeddings = self.generate_documents(100, 384)
            store.add_documents(docs, embeddings)
            
            # 模拟并发搜索
            concurrent_times = []
            
            for i in range(concurrent_operations):
                query_embedding = embeddings[i % len(embeddings)]
                
                start_time = time.time()
                results = store.search(query_embedding, top_k=5)
                search_time = time.time() - start_time
                
                concurrent_times.append(search_time)
                result.add_timing("concurrent_search", search_time)
            
            result.add_metric("concurrent_operations", concurrent_operations, "operations")
            
        finally:
            store.close()
        
        result.calculate_stats()
        return result
    
    def generate_report(self, results: List[BenchmarkResult], output_file: str = None) -> Dict[str, Any]:
        """生成基准测试报告"""
        report = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "results": [result.to_dict() for result in results],
            "summary": {}
        }
        
        # 计算汇总统计
        if results:
            # 按存储类型分组
            by_type = {}
            for result in results:
                store_type = result.name.split("_")[0]
                if store_type not in by_type:
                    by_type[store_type] = []
                by_type[store_type].append(result)
            
            # 生成比较
            comparison = {}
            for store_type, type_results in by_type.items():
                if type_results:
                    # 计算平均指标
                    avg_metrics = {}
                    for result in type_results:
                        for metric, data in result.metrics.items():
                            if isinstance(data, dict) and "mean" in data:
                                if metric not in avg_metrics:
                                    avg_metrics[metric] = []
                                avg_metrics[metric].append(data["mean"])
                    
                    # 计算总体平均值
                    for metric, values in avg_metrics.items():
                        if values:
                            avg_metrics[metric] = {
                                "mean": statistics.mean(values),
                                "std": statistics.stdev(values) if len(values) > 1 else 0,
                                "count": len(values)
                            }
                    
                    comparison[store_type] = avg_metrics
            
            report["summary"]["comparison"] = comparison
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report


@pytest.mark.benchmark
class TestVectorStorageBenchmark:
    """向量存储基准测试类"""
    
    @pytest.fixture
    def benchmark_suite(self):
        """创建基准测试套件"""
        suite = VectorStorageBenchmark()
        yield suite
        suite.cleanup()
    
    @pytest.fixture
    def store_configs(self):
        """存储配置列表"""
        configs = [
            {
                "type": "memory",
                "vector_dimension": 384,
                "similarity": "dot_product"
            }
        ]
        
        # 如果FAISS可用，添加FAISS配置
        try:
            import faiss
            configs.append({
                "type": "faiss", 
                "vector_dimension": 384,
                "similarity": "dot_product",
                "faiss": {
                    "index_type": "Flat",
                    "auto_save_interval": 0
                },
                "test_persistence": True
            })
        except ImportError:
            pass
        
        return configs
    
    def test_basic_operations_benchmark(self, benchmark_suite, store_configs):
        """基础操作基准测试"""
        test_params = {
            "document_count": 1000,
            "vector_dimension": 384,
            "batch_size": 100,
            "search_iterations": 20,
            "top_k": 10,
            "content_size": "medium"
        }
        
        results = []
        
        for config in store_configs:
            print(f"\nBenchmarking {config['type']} store...")
            result = benchmark_suite.benchmark_store_operations(config, test_params)
            results.append(result)
            
            # 输出关键指标
            metrics = result.metrics
            if "docs_per_second" in metrics:
                print(f"  Documents per second: {metrics['docs_per_second']['value']:.2f}")
            if "avg_search_time" in metrics:
                print(f"  Average search time: {metrics['avg_search_time']['value']*1000:.2f}ms")
        
        # 验证所有基准测试都成功
        for result in results:
            assert "error" not in result.metrics
            assert result.metrics.get("final_document_count", {}).get("value", 0) > 0
        
        benchmark_suite.results.extend(results)
    
    @pytest.mark.slow
    def test_scalability_benchmark(self, benchmark_suite, store_configs):
        """可扩展性基准测试"""
        document_counts = [100, 500, 1000, 2000]
        
        results = benchmark_suite.run_scalability_test(store_configs, document_counts)
        
        # 验证扩展性趋势
        for config in store_configs:
            store_results = [r for r in results if r.name.startswith(config["type"])]
            
            # 检查随着文档数量增加的性能变化
            add_times = []
            search_times = []
            
            for result in sorted(store_results, key=lambda x: int(x.name.split("_")[-2])):
                if "total_add_time" in result.metrics:
                    add_times.append(result.metrics["total_add_time"]["value"])
                if "avg_search_time" in result.metrics:
                    search_times.append(result.metrics["avg_search_time"]["value"])
            
            # 验证性能指标存在且合理
            assert len(add_times) > 0, f"No add time metrics for {config['type']}"
            assert all(t > 0 for t in add_times), f"Invalid add times for {config['type']}"
            
            if search_times:
                assert all(t >= 0 for t in search_times), f"Invalid search times for {config['type']}"
        
        benchmark_suite.results.extend(results)
    
    def test_vector_dimension_impact_benchmark(self, benchmark_suite, store_configs):
        """向量维度影响基准测试"""
        dimensions = [128, 256, 384, 512]
        
        results = benchmark_suite.run_dimension_impact_test(store_configs, dimensions)
        
        # 验证维度影响
        for config in store_configs:
            dim_results = [r for r in results if r.name.startswith(config["type"])]
            
            # 检查不同维度的性能差异
            for result in dim_results:
                assert "error" not in result.metrics
                assert result.metrics.get("final_document_count", {}).get("value", 0) > 0
        
        benchmark_suite.results.extend(results)
    
    def test_concurrent_access_benchmark(self, benchmark_suite, store_configs):
        """并发访问基准测试"""
        for config in store_configs:
            result = benchmark_suite.run_concurrent_access_test(config, concurrent_operations=10)
            
            # 验证并发测试结果
            assert "error" not in result.metrics
            assert result.metrics.get("concurrent_operations", {}).get("value", 0) == 10
            
            if "concurrent_search" in result.metrics and "mean" in result.metrics["concurrent_search"]:
                avg_time = result.metrics["concurrent_search"]["mean"]
                assert avg_time > 0, f"Invalid concurrent search time for {config['type']}"
            
            benchmark_suite.results.append(result)
    
    def test_memory_usage_benchmark(self, benchmark_suite, store_configs):
        """内存使用基准测试"""
        # 注意：这是一个简化的内存测试
        # 实际应用中可能需要使用psutil等工具监控内存使用
        
        import sys
        
        for config in store_configs:
            # 记录初始内存使用（简化）
            initial_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
            
            # 执行基准测试
            test_params = {
                "document_count": 1000,
                "vector_dimension": 384,
                "batch_size": 100,
                "search_iterations": 5,
                "top_k": 5
            }
            
            result = benchmark_suite.benchmark_store_operations(config, test_params)
            
            # 记录最终内存使用（简化）
            final_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
            
            result.add_metric("object_count_increase", final_objects - initial_objects, "objects")
            
            benchmark_suite.results.append(result)
    
    def test_generate_comprehensive_report(self, benchmark_suite):
        """生成综合基准报告"""
        if not benchmark_suite.results:
            # 如果没有其他测试结果，运行一个基础测试
            config = {"type": "memory", "vector_dimension": 384}
            test_params = {
                "document_count": 100,
                "vector_dimension": 384,
                "batch_size": 50,
                "search_iterations": 5,
                "top_k": 5
            }
            
            result = benchmark_suite.benchmark_store_operations(config, test_params)
            benchmark_suite.results.append(result)
        
        # 生成报告
        report = benchmark_suite.generate_report(benchmark_suite.results)
        
        # 验证报告结构
        assert "benchmark_timestamp" in report
        assert "results" in report
        assert "summary" in report
        assert len(report["results"]) > 0
        
        # 可选：保存报告到文件
        output_file = os.path.join(tempfile.gettempdir(), "vector_storage_benchmark_report.json")
        benchmark_suite.generate_report(benchmark_suite.results, output_file)
        
        assert os.path.exists(output_file)
        print(f"\nBenchmark report saved to: {output_file}")


# 导入gc用于内存测试
import gc


if __name__ == "__main__":
    # 运行特定的基准测试
    pytest.main([__file__, "-v", "-m", "benchmark", "--tb=short"])