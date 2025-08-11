"""
Performance Enhancement Utilities for MCP Academic RAG Server

This module provides production-ready performance optimizations including:
- Async/await improvements for I/O operations
- Memory management and monitoring
- Connection pooling and resource management
- Intelligent caching systems
- Performance profiling and metrics
- Batch processing optimization
"""

import asyncio
import aiofiles
import aiohttp
import time
import gc
import logging
import threading
from typing import Dict, Any, Optional, List, Callable, Union, AsyncGenerator
from functools import wraps, lru_cache
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import weakref
import json
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    success: bool
    error_message: Optional[str] = None


class MemoryManager:
    """Advanced memory management and monitoring"""
    
    def __init__(self):
        self.memory_threshold = 80.0  # 80% of available memory
        self.gc_threshold = 85.0  # Force GC at 85%
        self.metrics_history = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),  # GB
            'percent': memory.percent,
            'used': memory.used / (1024**3)  # GB
        }
    
    @classmethod
    def check_memory_threshold(cls, threshold: float = 80.0) -> bool:
        """Check if memory usage exceeds threshold"""
        return cls.get_memory_usage()['percent'] > threshold
    
    @staticmethod
    def collect_garbage() -> Dict[str, int]:
        """Force garbage collection and return statistics"""
        before_objects = len(gc.get_objects())
        collected = gc.collect()
        after_objects = len(gc.get_objects())
        
        return {
            'collected': collected,
            'objects_before': before_objects,
            'objects_after': after_objects,
            'objects_freed': before_objects - after_objects
        }
    
    def monitor_memory(self, operation_name: str = "unknown"):
        """Context manager for monitoring memory usage during operations"""
        @contextmanager
        def memory_monitor():
            memory_before = self.get_memory_usage()
            start_time = time.time()
            
            try:
                yield memory_before
            finally:
                end_time = time.time()
                memory_after = self.get_memory_usage()
                
                # Log memory usage if significant increase
                memory_increase = memory_after['used'] - memory_before['used']
                if memory_increase > 0.1:  # More than 100MB increase
                    logger.warning(f"Operation '{operation_name}' increased memory by {memory_increase:.2f}GB")
                
                # Force GC if memory usage is high
                if memory_after['percent'] > self.gc_threshold:
                    logger.info("High memory usage detected, forcing garbage collection")
                    gc_stats = self.collect_garbage()
                    logger.info(f"Garbage collection freed {gc_stats['objects_freed']} objects")
                
                # Store metrics
                with self._lock:
                    self.metrics_history.append({
                        'operation': operation_name,
                        'timestamp': end_time,
                        'duration': end_time - start_time,
                        'memory_before': memory_before['used'],
                        'memory_after': memory_after['used'],
                        'memory_increase': memory_increase
                    })
        
        return memory_monitor()


class ConnectionPool:
    """Generic connection pool for resource management"""
    
    def __init__(self, factory: Callable, max_size: int = 10, min_size: int = 2, 
                 timeout: float = 30.0, max_idle_time: float = 300.0):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.timeout = timeout
        self.max_idle_time = max_idle_time
        
        self._pool = deque()
        self._in_use = set()
        self._lock = asyncio.Lock()
        self._created_count = 0
        self._last_cleanup = time.time()
        
        # Weak references to track connection lifecycle
        self._connection_refs = weakref.WeakSet()
    
    async def _create_connection(self):
        """Create a new connection"""
        try:
            connection = await asyncio.to_thread(self.factory) if asyncio.iscoroutinefunction(self.factory) else self.factory()
            self._created_count += 1
            self._connection_refs.add(connection)
            logger.debug(f"Created new connection (total: {self._created_count})")
            return connection
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise
    
    async def get_connection(self):
        """Get a connection from the pool"""
        async with self._lock:
            # Try to get existing connection from pool
            while self._pool:
                connection, created_time = self._pool.popleft()
                
                # Check if connection is still valid and not too old
                if time.time() - created_time < self.max_idle_time:
                    self._in_use.add(connection)
                    return connection
                else:
                    # Connection is too old, discard it
                    logger.debug("Discarding old connection from pool")
            
            # No available connections, create new one if under limit
            if len(self._in_use) < self.max_size:
                connection = await self._create_connection()
                self._in_use.add(connection)
                return connection
            
            # Pool is full, wait or raise exception
            raise Exception(f"Connection pool exhausted (max_size: {self.max_size})")
    
    async def return_connection(self, connection):
        """Return a connection to the pool"""
        async with self._lock:
            if connection in self._in_use:
                self._in_use.remove(connection)
                
                # Return to pool if under max size
                if len(self._pool) < self.max_size:
                    self._pool.append((connection, time.time()))
                else:
                    # Pool is full, close the connection
                    await self._close_connection(connection)
    
    async def _close_connection(self, connection):
        """Close a connection"""
        try:
            if hasattr(connection, 'close'):
                if asyncio.iscoroutinefunction(connection.close):
                    await connection.close()
                else:
                    connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    async def cleanup(self):
        """Clean up old connections"""
        current_time = time.time()
        if current_time - self._last_cleanup < 60:  # Cleanup every minute
            return
        
        async with self._lock:
            # Remove old connections from pool
            active_connections = deque()
            while self._pool:
                connection, created_time = self._pool.popleft()
                if current_time - created_time < self.max_idle_time:
                    active_connections.append((connection, created_time))
                else:
                    await self._close_connection(connection)
            
            self._pool = active_connections
            self._last_cleanup = current_time
    
    @asynccontextmanager
    async def connection(self):
        """Context manager for getting and returning connections"""
        conn = await self.get_connection()
        try:
            yield conn
        finally:
            await self.return_connection(conn)


class CacheManager:
    """Multi-level caching system with TTL support"""
    
    def __init__(self, backend: str = "memory", max_size: int = 1000, 
                 use_ttl: bool = True, default_ttl: int = 3600):
        self.backend = backend
        self.max_size = max_size
        self.use_ttl = use_ttl
        self.default_ttl = default_ttl
        
        if backend == "redis" and REDIS_AVAILABLE:
            self.redis_client = self._create_redis_client()
        else:
            self.redis_client = None
            # Memory cache implementation
            self._memory_cache = {}
            self._access_times = {}
            self._ttl_cache = {}
            self._lock = threading.Lock()
    
    def _create_redis_client(self):
        """Create Redis client"""
        try:
            redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
            return redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            return None
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        import hashlib
        
        # Create a string representation of arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        args_hash = hashlib.md5(args_str.encode()).hexdigest()
        
        return f"{func_name}:{args_hash}"
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if not self.use_ttl or key not in self._ttl_cache:
            return False
        
        return time.time() > self._ttl_cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                logger.warning(f"Redis cache get error: {e}")
                return None
        else:
            with self._lock:
                if key in self._memory_cache and not self._is_expired(key):
                    self._access_times[key] = time.time()
                    return self._memory_cache[key]
                elif key in self._memory_cache:
                    # Expired entry, remove it
                    del self._memory_cache[key]
                    del self._ttl_cache[key]
                    del self._access_times[key]
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        else:
            with self._lock:
                # Evict old entries if cache is full
                if len(self._memory_cache) >= self.max_size:
                    self._evict_lru()
                
                self._memory_cache[key] = value
                self._access_times[key] = time.time()
                if self.use_ttl:
                    self._ttl_cache[key] = time.time() + ttl
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times, key=self._access_times.get)
        del self._memory_cache[lru_key]
        del self._access_times[lru_key]
        if lru_key in self._ttl_cache:
            del self._ttl_cache[lru_key]
    
    def delete(self, key: str) -> None:
        """Delete entry from cache"""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis cache delete error: {e}")
        else:
            with self._lock:
                self._memory_cache.pop(key, None)
                self._access_times.pop(key, None)
                self._ttl_cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache clear error: {e}")
        else:
            with self._lock:
                self._memory_cache.clear()
                self._access_times.clear()
                self._ttl_cache.clear()


def cached(cache_manager: CacheManager, ttl: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_manager._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss for {func.__name__}, executing function")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        # For async functions
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = cache_manager._generate_key(func.__name__, args, kwargs)
            
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            logger.debug(f"Cache miss for {func.__name__}, executing function")
            result = await func(*args, **kwargs)
            
            cache_manager.set(cache_key, result, ttl)
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


class BatchProcessor:
    """Intelligent batch processing with dynamic optimization"""
    
    def __init__(self, batch_size: int = 32, timeout: float = 5.0, 
                 process_func: Callable = None, max_retries: int = 3):
        self.batch_size = batch_size
        self.timeout = timeout
        self.process_func = process_func
        self.max_retries = max_retries
        
        self._batch = []
        self._lock = asyncio.Lock()
        self._last_batch_time = time.time()
        self._processing = False
        
        # Performance metrics
        self._batch_times = deque(maxlen=100)
        self._optimal_batch_size = batch_size
    
    async def add(self, item: Any) -> Optional[Any]:
        """Add item to batch for processing"""
        async with self._lock:
            self._batch.append(item)
            
            # Process batch if size threshold reached or timeout exceeded
            should_process = (
                len(self._batch) >= self.batch_size or
                time.time() - self._last_batch_time > self.timeout
            )
            
            if should_process and not self._processing:
                return await self._process_batch()
    
    async def flush(self) -> Optional[Any]:
        """Force process current batch"""
        async with self._lock:
            if self._batch and not self._processing:
                return await self._process_batch()
    
    async def _process_batch(self) -> Optional[Any]:
        """Process current batch"""
        if not self._batch or not self.process_func:
            return None
        
        self._processing = True
        batch_to_process = self._batch.copy()
        self._batch.clear()
        self._last_batch_time = time.time()
        
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(self.process_func):
                result = await self.process_func(batch_to_process)
            else:
                result = await asyncio.to_thread(self.process_func, batch_to_process)
            
            processing_time = time.time() - start_time
            self._batch_times.append(processing_time)
            
            # Optimize batch size based on processing time
            self._optimize_batch_size()
            
            logger.debug(f"Processed batch of {len(batch_to_process)} items in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Could implement retry logic here
            return None
        finally:
            self._processing = False
    
    def _optimize_batch_size(self):
        """Dynamically optimize batch size based on processing times"""
        if len(self._batch_times) < 10:
            return
        
        avg_time = sum(self._batch_times) / len(self._batch_times)
        
        # Target processing time of 2 seconds
        target_time = 2.0
        
        if avg_time < target_time * 0.8:  # Processing too fast, increase batch size
            self._optimal_batch_size = min(self._optimal_batch_size * 1.2, 1000)
        elif avg_time > target_time * 1.2:  # Processing too slow, decrease batch size
            self._optimal_batch_size = max(self._optimal_batch_size * 0.8, 10)
        
        self.batch_size = int(self._optimal_batch_size)
        logger.debug(f"Optimized batch size to {self.batch_size}")


def profile_performance(operation_name: str = "unknown"):
    """Performance profiling decorator and context manager"""
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            memory_manager = MemoryManager()
            start_time = time.time()
            memory_before = memory_manager.get_memory_usage()
            success = False
            error_message = None
            
            try:
                with memory_manager.monitor_memory(f"{operation_name}-{func.__name__}"):
                    result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_message = str(e)
                logger.error(f"Performance profiled function {func.__name__} failed: {e}")
                raise
            finally:
                end_time = time.time()
                memory_after = memory_manager.get_memory_usage()
                
                metrics = PerformanceMetrics(
                    operation_name=f"{operation_name}-{func.__name__}",
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    memory_before=memory_before['used'],
                    memory_after=memory_after['used'],
                    success=success,
                    error_message=error_message
                )
                
                # Log performance metrics
                logger.info(
                    f"Performance: {metrics.operation_name} - "
                    f"Duration: {metrics.duration:.3f}s, "
                    f"Memory: {metrics.memory_after - metrics.memory_before:.2f}GB, "
                    f"Success: {success}"
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            memory_manager = MemoryManager()
            start_time = time.time()
            memory_before = memory_manager.get_memory_usage()
            success = False
            error_message = None
            
            try:
                with memory_manager.monitor_memory(f"{operation_name}-{func.__name__}"):
                    result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_message = str(e)
                logger.error(f"Performance profiled function {func.__name__} failed: {e}")
                raise
            finally:
                end_time = time.time()
                memory_after = memory_manager.get_memory_usage()
                
                logger.info(
                    f"Performance: {operation_name}-{func.__name__} - "
                    f"Duration: {end_time - start_time:.3f}s, "
                    f"Memory: {memory_after['used'] - memory_before['used']:.2f}GB, "
                    f"Success: {success}"
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    # Can be used as decorator or context manager
    if callable(operation_name):
        func = operation_name
        operation_name = func.__name__
        return decorator(func)
    
    return decorator


async def optimize_batch_size(items: List[Any], process_func: Callable, target_time: float = 2.0) -> int:
    """Determine optimal batch size for processing function"""
    test_sizes = [1, 5, 10, 20, 50, 100]
    times = []
    
    for size in test_sizes:
        if size > len(items):
            break
        
        test_batch = items[:size]
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(process_func):
                await process_func(test_batch)
            else:
                await asyncio.to_thread(process_func, test_batch)
        except Exception as e:
            logger.warning(f"Batch size optimization test failed for size {size}: {e}")
            continue
        
        processing_time = time.time() - start_time
        times.append((size, processing_time))
        
        # Stop if we've found a good size
        if processing_time > target_time * 1.5:
            break
    
    if not times:
        return 32  # Default fallback
    
    # Find the size that gets closest to target time
    best_size = 32
    best_diff = float('inf')
    
    for size, time_taken in times:
        time_per_item = time_taken / size
        projected_time = time_per_item * (target_time / time_per_item)
        diff = abs(projected_time - target_time)
        
        if diff < best_diff:
            best_diff = diff
            best_size = int(target_time / time_per_item)
    
    return max(1, min(best_size, 1000))  # Clamp between 1 and 1000


# Export main classes and functions
__all__ = [
    'MemoryManager',
    'ConnectionPool', 
    'CacheManager',
    'BatchProcessor',
    'cached',
    'profile_performance',
    'optimize_batch_size',
    'PerformanceMetrics'
]