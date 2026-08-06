# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from time import sleep
from typing import Any

import pytest

from ess.reduce.data import Registry


class _CountingRegistry(Registry):
    def __init__(self) -> None:
        super().__init__({'file': 'md5:unused'})
        self.fetch_count = 0
        self._fetch_count_lock = Lock()

    def _fetch(self, name: str, extractor: Any) -> list[str]:
        with self._fetch_count_lock:
            self.fetch_count += 1
        # Keep the fetch in progress so concurrent callers observe a cache miss.
        sleep(0.05)
        return [name]

    def _extractor_processor(self, name: str) -> object:
        return object()


class _ConcurrentRegistry(Registry):
    def __init__(self) -> None:
        super().__init__(
            {
                'first': 'md5:unused',
                'second': 'md5:unused',
            }
        )
        self._fetch_barrier = Barrier(2)

    def _fetch(self, name: str, extractor: Any) -> list[str]:
        # Both fetches must be in progress for either one to complete.
        self._fetch_barrier.wait(timeout=2.0)
        return [name]

    def _extractor_processor(self, name: str) -> None:
        return None


@pytest.mark.parametrize('method_name', ['get_path', 'get_paths'])
def test_registry_fetches_file_once_for_concurrent_access(method_name: str) -> None:
    registry = _CountingRegistry()
    worker_count = 4
    start = Barrier(worker_count)

    def access_file(_: int) -> None:
        start.wait()
        getattr(registry, method_name)('file')

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(access_file, range(worker_count)))

    assert registry.fetch_count == 1


def test_registry_fetches_different_files_concurrently() -> None:
    registry = _ConcurrentRegistry()

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(executor.map(registry.get_path, ['first', 'second']))

    assert paths == [Path('first'), Path('second')]
