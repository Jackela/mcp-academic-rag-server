"""Owned connection/cache resources must work for actual async factories and failed optional clients."""

from unittest.mock import patch

import pytest

from utils.performance_enhancements import CacheManager, ConnectionPool, profile_performance


class Connection:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_async_factory_is_awaited_and_returned_resource_reused():
    resource = Connection()

    async def factory():
        return resource

    pool = ConnectionPool(factory, max_size=1)
    async with pool.connection() as actual:
        assert actual is resource
    assert await pool.get_connection() is resource
    await pool.return_connection(resource)


@pytest.mark.asyncio
async def test_expired_owned_connection_is_closed_before_replacement():
    resources = []

    def factory():
        resource = Connection()
        resources.append(resource)
        return resource

    pool = ConnectionPool(factory, max_size=1, max_idle_time=1)
    with patch("utils.performance_enhancements.time.time", return_value=100):
        first = await pool.get_connection()
        await pool.return_connection(first)
    with patch("utils.performance_enhancements.time.time", return_value=102):
        second = await pool.get_connection()
    assert first.closed and not second.closed
    assert first is not second
    await pool.return_connection(second)


def test_failed_optional_redis_creation_keeps_functional_memory_cache():
    with (
        patch("utils.performance_enhancements.REDIS_AVAILABLE", True),
        patch.object(CacheManager, "_create_redis_client", return_value=None),
    ):
        cache = CacheManager("redis", max_size=2)
    cache.set("owned", 42)
    assert cache.get("owned") == 42
    cache.clear()
    assert cache.get("owned") is None


def test_profiling_direct_decorator_preserves_errors_and_return_values():
    @profile_performance
    def direct(value):
        if value < 0:
            raise ValueError("controlled negative")
        return value + 1

    assert direct(2) == 3
    with pytest.raises(ValueError, match="controlled negative"):
        direct(-1)
