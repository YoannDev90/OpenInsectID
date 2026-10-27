"""Run coroutines from synchronous code, including inside Jupyter.

``asyncio.run`` raises ``RuntimeError: asyncio.run() cannot be called from a
running event loop`` under ipykernel (the kernel already runs an event loop in
the calling thread), which is exactly where the Kaggle notebooks execute.
The fix: fall back to a worker thread with a fresh loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, TypeVar

T = TypeVar("T")


def run_async(coro: Awaitable[T]) -> T:
    """Await ``coro`` to completion, standalone or nested in a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Inside an event loop (Jupyter): block this thread while a worker thread
    # runs the coroutine on a private loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
