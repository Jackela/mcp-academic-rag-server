"""
Performance optimization utilities for the Academic RAG Server

This module provides performance optimization features including:
- Caching mechanisms
- Connection pooling
- Batch processing
- Memory management
- Performance profiling
"""

import time
import asyncio
import functools
import hashlib
import json
import pickle
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import threading
import psutil
import gc
from dataclasses import dataclass
from enum import Enum

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import memcache
    HAS_MEMCACHE = True
except ImportError:
    HAS_MEMCACHE = False

logger = logging.getLogger(__name__)
T = TypeVar('T')


class CacheBackend(Enum):
    """Cache backend types"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHE = "memcache"
    FILE = "file"


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats.hits += 1
                return self.cache[key]
            else:
                self.stats.misses += 1
                return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Evict least recently used
                    self.cache.popitem(last=False)
                    self.stats.evictions += 1
            
            self.cache[key] = value
            self.stats.size = len(self.cache)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.size = len(self.cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats


class TTLCache:
    """Time-based cache with TTL support"""
    
    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    self.stats.hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
                    self.stats.size = len(self.cache)
            
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with TTL"""
        with self.lock:
            ttl = ttl or self.default_ttl
            expiry = time.time() + ttl
            
            # Check size limit
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
                self.stats.evictions += 1
            
            self.cache[key] = (value, expiry)
            self.stats.size = len(self.cache)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                k for k, (_, expiry) in self.cache.items()
                if current_time >= expiry
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            self.stats.size = len(self.cache)
            return len(expired_keys)


class CacheManager:
    """Unified cache manager supporting multiple backends"""
    
    def __init__(
        self,
        backend: CacheBackend = CacheBackend.MEMORY,
        **kwargs
    ):
        self.backend = backend
        self._cache = self._init_backend(**kwargs)
    
    def _init_backend(self, **kwargs):
        """Initialize cache backend"""
        if self.backend == CacheBackend.MEMORY:
            max_size = kwargs.get('max_size', 1000)
            if kwargs.get('use_ttl', False):
                return TTLCache(
                    default_ttl=kwargs.get('default_ttl', 3600),
                    max_size=max_size
                )
            else:
                return LRUCache(max_size=max_size)
        
        elif self.backend == CacheBackend.REDIS and HAS_REDIS:
            return redis.Redis(
                host=kwargs.get('host', 'localhost'),
                port=kwargs.get('port', 6379),
                db=kwargs.get('db', 0),
                decode_responses=True
            )
        
        elif self.backend == CacheBackend.MEMCACHE and HAS_MEMCACHE:
            servers = kwargs.get('servers', ['127.0.0.1:11211'])
            return memcache.Client(servers)
        
        else:
            # Fallback to memory cache
            logger.warning(f"Backend {self.backend} not available, using memory cache")
            return LRUCache(kwargs.get('max_size', 1000))
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if hasattr(self._cache, 'get'):
                return self._cache.get(key)
            else:
                # Redis/Memcache
                value = self._cache.get(key)
                if value and isinstance(value, bytes):
                    return pickle.loads(value)
                return value
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            if hasattr(self._cache, 'set'):
                if isinstance(self._cache, TTLCache):
                    self._cache.set(key, value, ttl)
                else:
                    self._cache.set(key, value)
                return True
            else:
                # Redis/Memcache
                serialized = pickle.dumps(value)
                if ttl:
                    return self._cache.setex(key, ttl, serialized)
                else:
                    return self._cache.set(key, serialized)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False


def cached(
    cache_manager: Optional[CacheManager] = None,
    key_func: Optional[Callable] = None,
    ttl: Optional[int] = None
):
    """
    Decorator for caching function results
    
    Args:
        cache_manager: Cache manager instance
        key_func: Function to generate cache key
        ttl: Time to live in seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Use default cache if not provided
        _cache = cache_manager or CacheManager()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(func, *args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(
                    ":".join(key_parts).encode()
                ).hexdigest()
            
            # Try to get from cache
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            _cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}, stored result")
            
            return result
        
        return wrapper
    return decorator


def async_cached(
    cache_manager: Optional[CacheManager] = None,
    key_func: Optional[Callable] = None,
    ttl: Optional[int] = None
):
    """Async version of cached decorator"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        _cache = cache_manager or CacheManager()
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(func, *args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(
                    ":".join(key_parts).encode()
                ).hexdigest()
            
            # Try to get from cache
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute async function
            result = await func(*args, **kwargs)
            
            # Store in cache
            _cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class ConnectionPool:
    """Generic connection pool implementation"""
    
    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = 10,
        min_size: int = 2,
        timeout: float = 30.0
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.timeout = timeout
        
        self._pool: List[Any] = []
        self._in_use: Set[Any] = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
        # Pre-create minimum connections
        for _ in range(self.min_size):
            conn = self.factory()
            self._pool.append(conn)
    
    def acquire(self) -> Any:
        """Acquire connection from pool"""
        deadline = time.time() + self.timeout
        
        with self._condition:
            while True:
                # Try to get existing connection
                if self._pool:
                    conn = self._pool.pop()
                    self._in_use.add(conn)
                    return conn
                
                # Create new connection if under limit
                if len(self._in_use) < self.max_size:
                    conn = self.factory()
                    self._in_use.add(conn)
                    return conn
                
                # Wait for connection to be released
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Connection pool timeout")
                
                self._condition.wait(timeout=remaining)
    
    def release(self, conn: Any) -> None:
        """Release connection back to pool"""
        with self._condition:
            if conn in self._in_use:
                self._in_use.remove(conn)
                self._pool.append(conn)
                self._condition.notify()
    
    @contextmanager
    def connection(self):
        """Context manager for connection"""
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)


class BatchProcessor:
    """Batch processing utilities"""
    
    def __init__(
        self,
        batch_size: int = 100,
        timeout: float = 1.0,
        process_func: Optional[Callable[[List[Any]], None]] = None
    ):
        self.batch_size = batch_size
        self.timeout = timeout
        self.process_func = process_func
        
        self._batch: List[Any] = []
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
    
    def add(self, item: Any) -> None:
        """Add item to batch"""
        with self._lock:
            self._batch.append(item)
            
            # Process if batch is full
            if len(self._batch) >= self.batch_size:
                self._process_batch()
            else:
                # Start timer if not already running
                if not self._timer or not self._timer.is_alive():
                    self._timer = threading.Timer(
                        self.timeout,
                        self._process_batch
                    )
                    self._timer.start()
    
    def _process_batch(self) -> None:
        """Process current batch"""
        with self._lock:
            if not self._batch:
                return
            
            # Cancel timer if running
            if self._timer and self._timer.is_alive():
                self._timer.cancel()
            
            # Process batch
            if self.process_func:
                try:
                    self.process_func(self._batch.copy())
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
            
            # Clear batch
            self._batch.clear()
    
    def flush(self) -> None:
        """Force process remaining items"""
        self._process_batch()


class MemoryManager:
    """Memory usage monitoring and management"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss,  # Resident Set Size
            'vms': memory_info.vms,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available
        }
    
    @staticmethod
    def check_memory_threshold(threshold_percent: float = 80.0) -> bool:
        """Check if memory usage exceeds threshold"""
        usage = MemoryManager.get_memory_usage()
        return usage['percent'] > threshold_percent
    
    @staticmethod
    def collect_garbage() -> Dict[str, int]:
        """Force garbage collection and return statistics"""
        before = MemoryManager.get_memory_usage()
        
        # Collect garbage
        collected = {
            'generation_0': gc.collect(0),
            'generation_1': gc.collect(1),
            'generation_2': gc.collect(2)
        }
        
        after = MemoryManager.get_memory_usage()
        
        return {
            'collected': collected,
            'memory_freed': before['rss'] - after['rss']
        }


@contextmanager
def profile_performance(name: str = "Operation"):
    """Context manager for performance profiling"""
    start_time = time.time()
    start_memory = MemoryManager.get_memory_usage()
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = MemoryManager.get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory['rss'] - start_memory['rss']
        
        logger.info(
            f"{name} completed - "
            f"Duration: {duration:.3f}s, "
            f"Memory delta: {memory_delta / 1024 / 1024:.2f}MB"
        )


def optimize_batch_size(
    items: List[Any],
    process_func: Callable[[List[Any]], float],
    min_batch: int = 10,
    max_batch: int = 1000,
    target_time: float = 1.0
) -> int:
    """
    Find optimal batch size for processing
    
    Args:
        items: Sample items to test
        process_func: Function that processes batch and returns time taken
        min_batch: Minimum batch size
        max_batch: Maximum batch size
        target_time: Target processing time per batch
        
    Returns:
        Optimal batch size
    """
    best_batch_size = min_batch
    best_score = float('inf')
    
    # Test different batch sizes
    for batch_size in range(min_batch, min(max_batch, len(items)), 10):
        batch = items[:batch_size]
        
        # Measure processing time
        try:
            process_time = process_func(batch)
            
            # Score based on distance from target time
            score = abs(process_time - target_time)
            
            if score < best_score:
                best_score = score
                best_batch_size = batch_size
            
            # If we're taking too long, don't test larger batches
            if process_time > target_time * 2:
                break
                
        except Exception as e:
            logger.error(f"Batch size optimization error: {e}")
            continue
    
    logger.info(f"Optimal batch size: {best_batch_size}")
    return best_batch_size


# Example usage
if __name__ == "__main__":
    # Test LRU cache
    cache = LRUCache(max_size=3)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    
    print(f"Cache hit rate: {cache.get_stats().hit_rate:.2%}")
    
    # Test cached decorator
    @cached(ttl=60)
    def expensive_function(x: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return x * x
    
    # First call - cache miss
    with profile_performance("First call"):
        result1 = expensive_function(5)
    
    # Second call - cache hit
    with profile_performance("Second call"):
        result2 = expensive_function(5)
    
    assert result1 == result2
    
    # Test connection pool
    def create_connection():
        return {"id": time.time()}
    
    pool = ConnectionPool(create_connection, max_size=5)
    
    with pool.connection() as conn:
        print(f"Got connection: {conn}")
    
    # Test memory management
    memory_stats = MemoryManager.get_memory_usage()
    print(f"Memory usage: {memory_stats['percent']:.1f}%")