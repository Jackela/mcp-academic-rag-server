"""Offline regressions for normal logging and synchronous batch boundaries."""

import asyncio
import logging
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from utils.error_handling import RetryConfig, async_retry_with_backoff, retry_with_backoff
from utils.image_utils import ImageUtils
from utils.logging_utils import log_async_performance, log_performance
from utils.performance_utils import BatchProcessor, CacheBackend, CacheManager


class RuntimeUtilityTests(unittest.TestCase):
    def test_performance_logging_preserves_result_and_original_exception(self):
        @log_performance
        def operation(fail=False):
            if fail:
                raise ValueError("original failure")
            return 42

        with self.assertLogs(__name__, level=logging.INFO) as captured:
            self.assertEqual(operation(), 42)
            with self.assertRaisesRegex(ValueError, "original failure"):
                operation(True)
        self.assertEqual(len(captured.records), 2)
        self.assertEqual(captured.records[0].function_module, __name__)

    def test_async_performance_logging_preserves_result(self):
        @log_async_performance
        async def operation():
            return "complete"

        with self.assertLogs(__name__, level=logging.INFO):
            self.assertEqual(asyncio.run(operation()), "complete")

    def test_full_batch_does_not_deadlock_and_processes_once(self):
        processed = []
        processor = BatchProcessor(batch_size=1, process_func=processed.append)
        worker = threading.Thread(target=lambda: processor.add("document"), daemon=True)
        worker.start()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive(), "Full batch deadlocked inside add")
        self.assertEqual(processed, [["document"]])
        processor.flush()
        self.assertEqual(processed, [["document"]])

    def test_batch_callback_can_enqueue_next_batch_without_losing_it(self):
        processed = []
        processor = BatchProcessor(batch_size=1)

        def process(items):
            processed.append(items)
            if items == ["first"]:
                processor.add("second")

        processor.process_func = process
        with patch("threading.Timer"):
            worker = threading.Thread(target=lambda: processor.add("first"), daemon=True)
            worker.start()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive(), "Callback could not enqueue work")
        self.assertEqual(processed, [["first"], ["second"]])


class ControlledRemoteCache:
    """Actual remote-cache byte/expiry boundary, with no network service."""

    def __init__(self):
        self.values = {}
        self.expiries = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        if not isinstance(value, bytes):
            raise TypeError("Remote cache accepts bytes")
        self.values[key] = value
        return True

    def setex(self, key, ttl, value):
        self.expiries[key] = ttl
        return self.set(key, value)


class RemoteCacheTests(unittest.TestCase):
    def test_remote_cache_serializes_objects_and_applies_expiry(self):
        remote = ControlledRemoteCache()
        with patch("utils.performance_utils.redis.Redis", return_value=remote):
            manager = CacheManager(CacheBackend.REDIS)
        self.assertTrue(manager.set("document", {"title": "local"}, ttl=12))
        self.assertEqual(remote.expiries, {"document": 12})
        self.assertEqual(manager.get("document"), {"title": "local"})
        self.assertIsNone(manager.get("missing"))


class AdditionalUtilityTests(unittest.TestCase):
    def test_memory_cache_keeps_objects_without_remote_serialization(self):
        for options in ({}, {"use_ttl": True}):
            manager = CacheManager(**options)
            value = {"content": "local"}
            self.assertTrue(manager.set("document", value, ttl=10))
            self.assertIs(manager.get("document"), value)

    def test_invalid_retry_attempts_are_rejected_before_execution(self):
        for decorator in (retry_with_backoff, async_retry_with_backoff):
            with self.assertRaisesRegex(ValueError, "max_attempts"):
                decorator(RetryConfig(max_attempts=0))

    def test_image_save_accepts_a_filename_in_the_current_directory(self):
        previous = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertTrue(ImageUtils.save_image(Image.new("RGB", (2, 2)), "actual.png"))
                with Image.open("actual.png") as saved:
                    self.assertEqual(saved.size, (2, 2))
            finally:
                os.chdir(previous)
