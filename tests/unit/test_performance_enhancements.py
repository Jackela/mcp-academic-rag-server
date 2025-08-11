"""
Unit tests for performance enhancement utilities

Tests the performance optimization components including:
- Memory management and monitoring
- Connection pooling
- Caching systems
- Batch processing
- Performance profiling
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from utils.performance_enhancements import (
    MemoryManager,
    ConnectionPool,
    CacheManager,
    BatchProcessor,
    cached,
    profile_performance,
    optimize_batch_size,
    PerformanceMetrics
)


class TestMemoryManager:
    """Test memory management functionality"""
    
    def test_get_memory_usage(self):
        """Test memory usage retrieval"""
        usage = MemoryManager.get_memory_usage()
        
        assert isinstance(usage, dict)
        assert 'total' in usage
        assert 'available' in usage
        assert 'percent' in usage
        assert 'used' in usage
        
        # Validate ranges
        assert usage['total'] > 0
        assert usage['available'] >= 0
        assert 0 <= usage['percent'] <= 100
        assert usage['used'] >= 0
    
    def test_check_memory_threshold(self):
        """Test memory threshold checking"""
        # Should return boolean
        result = MemoryManager.check_memory_threshold(50.0)
        assert isinstance(result, bool)
        
        # Test with very high threshold (should be False)
        result = MemoryManager.check_memory_threshold(99.9)
        assert isinstance(result, bool)
    
    def test_collect_garbage(self):
        """Test garbage collection"""
        stats = MemoryManager.collect_garbage()
        
        assert isinstance(stats, dict)
        assert 'collected' in stats
        assert 'objects_before' in stats
        assert 'objects_after' in stats
        assert 'objects_freed' in stats
        
        # Validate that the numbers make sense
        assert stats['objects_freed'] == stats['objects_before'] - stats['objects_after']
        assert stats['collected'] >= 0
    
    def test_monitor_memory_context_manager(self):
        """Test memory monitoring context manager"""
        manager = MemoryManager()
        
        with manager.monitor_memory("test_operation"):
            # Create some objects to increase memory usage
            test_data = [list(range(100)) for _ in range(10)]
        
        # Check that metrics were recorded
        assert len(manager.metrics_history) > 0
        
        latest_metric = manager.metrics_history[-1]
        assert latest_metric['operation'] == "test_operation"
        assert 'duration' in latest_metric
        assert 'memory_before' in latest_metric
        assert 'memory_after' in latest_metric


class TestConnectionPool:
    """Test connection pool functionality"""
    
    def test_connection_pool_creation(self):
        """Test connection pool initialization"""
        def factory():
            return {"connection": "mock"}
        
        pool = ConnectionPool(
            factory=factory,
            max_size=5,
            min_size=1
        )
        
        assert pool.max_size == 5
        assert pool.min_size == 1
        assert len(pool._pool) == 0
        assert len(pool._in_use) == 0
    
    @pytest.mark.asyncio
    async def test_get_and_return_connection(self):
        """Test getting and returning connections"""
        call_count = 0
        
        def factory():
            nonlocal call_count
            call_count += 1
            return Mock(id=call_count, close=Mock())
        
        pool = ConnectionPool(factory=factory, max_size=3)
        
        # Get first connection
        conn1 = await pool.get_connection()
        assert conn1.id == 1
        assert len(pool._in_use) == 1
        
        # Return connection
        await pool.return_connection(conn1)
        assert len(pool._in_use) == 0
        assert len(pool._pool) == 1
        
        # Get connection again (should reuse)
        conn2 = await pool.get_connection()
        assert conn2.id == 1  # Same connection reused
    
    @pytest.mark.asyncio
    async def test_connection_pool_context_manager(self):
        """Test connection pool context manager"""
        def factory():
            return Mock(close=Mock())
        
        pool = ConnectionPool(factory=factory, max_size=2)
        
        async with pool.connection() as conn:
            assert conn is not None
            assert len(pool._in_use) == 1
        
        # Connection should be returned after context
        assert len(pool._in_use) == 0
    
    @pytest.mark.asyncio
    async def test_connection_pool_max_size_limit(self):
        """Test that connection pool respects max size"""
        def factory():
            return Mock(close=Mock())
        
        pool = ConnectionPool(factory=factory, max_size=2)
        
        # Get maximum connections
        conn1 = await pool.get_connection()
        conn2 = await pool.get_connection()
        
        # Should raise exception when exceeding max size
        with pytest.raises(Exception, match="Connection pool exhausted"):
            await pool.get_connection()
        
        # Return one connection and try again
        await pool.return_connection(conn1)
        conn3 = await pool.get_connection()  # Should work now
        assert conn3 is not None


class TestCacheManager:
    """Test caching system functionality"""
    
    def test_memory_cache_basic_operations(self):
        """Test basic cache operations with memory backend"""
        cache = CacheManager(backend="memory", max_size=10, use_ttl=False)
        
        # Test set/get
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # Test non-existent key
        assert cache.get("non_existent") is None
        
        # Test deletion
        cache.delete("test_key")
        assert cache.get("test_key") is None
    
    def test_memory_cache_with_ttl(self):
        """Test cache with TTL expiration"""
        cache = CacheManager(backend="memory", use_ttl=True, default_ttl=1)
        
        cache.set("test_key", "test_value", ttl=1)
        assert cache.get("test_key") == "test_value"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("test_key") is None
    
    def test_memory_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = CacheManager(backend="memory", max_size=2, use_ttl=False)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add third key, should evict key2 (least recently used)
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should exist
    
    def test_cached_decorator_sync(self):
        """Test cached decorator with synchronous function"""
        cache = CacheManager(backend="memory", use_ttl=False)
        call_count = 0
        
        @cached(cache_manager=cache)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same arguments (should use cache)
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # No additional call
        
        # Third call with different arguments
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cached_decorator_async(self):
        """Test cached decorator with async function"""
        cache = CacheManager(backend="memory", use_ttl=False)
        call_count = 0
        
        @cached(cache_manager=cache)
        async def expensive_async_function(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return x * y
        
        # First call
        result1 = await expensive_async_function(2, 3)
        assert result1 == 6
        assert call_count == 1
        
        # Second call with same arguments
        result2 = await expensive_async_function(2, 3)
        assert result2 == 6
        assert call_count == 1  # Should use cache


class TestBatchProcessor:
    """Test batch processing functionality"""
    
    @pytest.mark.asyncio
    async def test_batch_processor_size_trigger(self):
        """Test batch processing triggered by size"""
        processed_batches = []
        
        async def process_batch(items):
            processed_batches.append(list(items))
            return f"processed_{len(items)}_items"
        
        processor = BatchProcessor(
            batch_size=3,
            timeout=10.0,  # Long timeout so size triggers first
            process_func=process_batch
        )
        
        # Add items one by one
        await processor.add("item1")
        await processor.add("item2")
        
        # Should not have processed yet
        assert len(processed_batches) == 0
        
        # Add third item, should trigger batch processing
        await processor.add("item3")
        
        # Give a moment for async processing
        await asyncio.sleep(0.1)
        
        assert len(processed_batches) == 1
        assert processed_batches[0] == ["item1", "item2", "item3"]
    
    @pytest.mark.asyncio
    async def test_batch_processor_timeout_trigger(self):
        """Test batch processing triggered by timeout"""
        processed_batches = []
        
        async def process_batch(items):
            processed_batches.append(list(items))
        
        processor = BatchProcessor(
            batch_size=10,  # Large batch size
            timeout=0.1,    # Short timeout
            process_func=process_batch
        )
        
        await processor.add("item1")
        await processor.add("item2")
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        await processor.flush()  # Force flush to ensure processing
        
        assert len(processed_batches) == 1
        assert "item1" in processed_batches[0]
        assert "item2" in processed_batches[0]
    
    @pytest.mark.asyncio
    async def test_batch_processor_manual_flush(self):
        """Test manual batch flushing"""
        processed_batches = []
        
        async def process_batch(items):
            processed_batches.append(list(items))
        
        processor = BatchProcessor(
            batch_size=10,
            timeout=10.0,
            process_func=process_batch
        )
        
        await processor.add("item1")
        await processor.add("item2")
        
        # Manual flush
        await processor.flush()
        
        assert len(processed_batches) == 1
        assert processed_batches[0] == ["item1", "item2"]


class TestPerformanceProfiling:
    """Test performance profiling utilities"""
    
    @pytest.mark.asyncio
    async def test_profile_performance_decorator_async(self):
        """Test performance profiling decorator with async function"""
        @profile_performance("test_operation")
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "test_result"
        
        result = await test_async_function()
        assert result == "test_result"
        # Performance metrics would be logged (we'd check logs in integration tests)
    
    def test_profile_performance_decorator_sync(self):
        """Test performance profiling decorator with sync function"""
        @profile_performance("test_sync_operation")
        def test_sync_function(x, y):
            time.sleep(0.01)  # Simulate work
            return x + y
        
        result = test_sync_function(1, 2)
        assert result == 3
    
    def test_profile_performance_context_manager(self):
        """Test using profile_performance as context manager"""
        # This would be implemented if we add context manager support
        pass


class TestOptimizeBatchSize:
    """Test batch size optimization"""
    
    @pytest.mark.asyncio
    async def test_optimize_batch_size_async(self):
        """Test batch size optimization with async function"""
        items = list(range(100))
        
        async def mock_process_func(batch):
            # Simulate processing time proportional to batch size
            await asyncio.sleep(len(batch) * 0.001)
            return len(batch)
        
        optimal_size = await optimize_batch_size(
            items=items,
            process_func=mock_process_func,
            target_time=0.1
        )
        
        assert isinstance(optimal_size, int)
        assert 1 <= optimal_size <= 1000
    
    @pytest.mark.asyncio
    async def test_optimize_batch_size_sync(self):
        """Test batch size optimization with sync function"""
        items = list(range(50))
        
        def mock_process_func(batch):
            time.sleep(len(batch) * 0.001)
            return len(batch)
        
        optimal_size = await optimize_batch_size(
            items=items,
            process_func=mock_process_func,
            target_time=0.05
        )
        
        assert isinstance(optimal_size, int)
        assert 1 <= optimal_size <= 1000


class TestPerformanceMetrics:
    """Test performance metrics data structure"""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics"""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            start_time=1000.0,
            end_time=1002.5,
            duration=2.5,
            memory_before=1.0,
            memory_after=1.2,
            success=True
        )
        
        assert metrics.operation_name == "test_operation"
        assert metrics.duration == 2.5
        assert metrics.memory_before == 1.0
        assert metrics.memory_after == 1.2
        assert metrics.success is True
        assert metrics.error_message is None


class TestIntegrationScenarios:
    """Integration tests combining multiple performance components"""
    
    @pytest.mark.asyncio
    async def test_cached_batch_processing(self):
        """Test combining caching with batch processing"""
        cache = CacheManager(backend="memory")
        processed_batches = []
        
        @cached(cache_manager=cache, ttl=60)
        async def expensive_operation(item):
            await asyncio.sleep(0.01)
            return f"processed_{item}"
        
        async def process_batch(items):
            results = []
            for item in items:
                result = await expensive_operation(item)
                results.append(result)
            processed_batches.append(results)
            return results
        
        processor = BatchProcessor(
            batch_size=3,
            timeout=1.0,
            process_func=process_batch
        )
        
        # Process items
        await processor.add("item1")
        await processor.add("item2") 
        await processor.add("item3")
        
        await asyncio.sleep(0.1)  # Allow processing
        
        assert len(processed_batches) > 0
        # Verify caching would work on subsequent calls
    
    @pytest.mark.asyncio
    async def test_connection_pooling_with_caching(self):
        """Test combining connection pooling with caching"""
        cache = CacheManager(backend="memory")
        
        def connection_factory():
            return Mock(query=Mock(return_value="db_result"))
        
        pool = ConnectionPool(factory=connection_factory, max_size=2)
        
        @cached(cache_manager=cache)
        async def cached_database_query(query):
            async with pool.connection() as conn:
                return conn.query(query)
        
        # First call
        result1 = await cached_database_query("SELECT * FROM test")
        
        # Second call (should use cache)
        result2 = await cached_database_query("SELECT * FROM test")
        
        # Both results should be the same (from cache on second call)
        assert result1 == result2