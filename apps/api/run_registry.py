"""
apps/api/run_registry.py
------------------------
Thread-safe registry for active run resources.

Each active run owns three optional resource slots:

* **task**     — the ``asyncio.Task`` driving the paper/live engine loop.
* **engine**   — a ``StrategyEngine`` instance (typed ``Any`` to avoid a
                 circular import from ``packages/trading``).
* **learning** — an ``AdaptiveLearningTask`` instance (optional, also ``Any``).

All multi-step mutations (``register``, ``unregister``, ``cancel_all``) execute
under a single ``asyncio.Lock`` so concurrent start/stop requests cannot
corrupt the internal dicts.  Read operations (``get_*``, ``active_run_ids``,
``__contains__``) are lock-free: Python's GIL makes individual dict-key
accesses atomic, and these paths are called per-bar where latency matters.

This module is intentionally side-effect-free on import so it can be
instantiated inside ``AppContainer`` (Stap 1b) without affecting the module
loader.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

__all__ = ["RunRegistry"]

logger = logging.getLogger(__name__)


class RunRegistry:
    """Thread-safe registry for active run resources (tasks, engines, learning instances).

    Usage::

        registry = RunRegistry()

        # On run start:
        await registry.register(run_id, task=t, engine=eng, learning=lt)

        # On run stop:
        await registry.unregister(run_id)

        # On API shutdown:
        await registry.cancel_all(timeout=5.0)
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._engines: dict[str, Any] = {}
        self._learning: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Mutating operations — executed under lock
    # ------------------------------------------------------------------

    async def register(
        self,
        run_id: str,
        task: asyncio.Task[Any],
        engine: Any,
        learning: Any | None = None,
    ) -> None:
        """Register all resources for a run atomically under a lock.

        Calling ``register`` for an already-registered *run_id* **overwrites**
        the previous entry.  Callers should ``unregister`` the old resources
        first (or cancel the old task) before registering a replacement.
        """
        async with self._lock:
            self._tasks[run_id] = task
            self._engines[run_id] = engine
            if learning is not None:
                self._learning[run_id] = learning
            elif run_id in self._learning:
                # Explicit None clears a previously registered learning task.
                del self._learning[run_id]

    async def unregister(self, run_id: str) -> None:
        """Remove all resources for a run atomically.

        Safe to call multiple times — subsequent calls for an already-removed
        *run_id* are silent no-ops.
        """
        async with self._lock:
            self._tasks.pop(run_id, None)
            self._engines.pop(run_id, None)
            self._learning.pop(run_id, None)

    async def cancel_all(self, timeout: float = 5.0) -> None:
        """Cancel every registered task and wait up to *timeout* seconds for cleanup.

        Internally:

        1. Acquires the lock to snapshot the current task dict.
        2. Cancels every non-done task.
        3. Releases the lock, then waits (via ``asyncio.wait_for``) for all
           gathered futures to settle within *timeout* seconds.
        4. Any ``CancelledError`` or other exception returned by a task is
           logged at DEBUG level and swallowed — shutdown must not raise.

        After this method returns the registry may still hold stale references
        if callers did not also call ``unregister``.  ``AppContainer`` is
        expected to call ``cancel_all`` once and then discard the registry.
        """
        async with self._lock:
            tasks = list(self._tasks.values())

        if not tasks:
            return

        pending = [t for t in tasks if not t.done()]
        for t in pending:
            t.cancel()

        if not pending:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "cancel_all timed out after %.1f s; %d task(s) may still be running",
                timeout,
                sum(1 for t in pending if not t.done()),
            )

    # ------------------------------------------------------------------
    # Read operations — lock-free
    # ------------------------------------------------------------------

    def get_task(self, run_id: str) -> asyncio.Task[Any] | None:
        """Return the task for *run_id*, or ``None`` if not registered."""
        return self._tasks.get(run_id)

    def get_engine(self, run_id: str) -> Any | None:
        """Return the engine for *run_id*, or ``None`` if not registered."""
        return self._engines.get(run_id)

    def get_learning(self, run_id: str) -> Any | None:
        """Return the learning task for *run_id*, or ``None`` if not registered."""
        return self._learning.get(run_id)

    def active_run_ids(self) -> list[str]:
        """Return a snapshot list of currently registered run IDs.

        Modifying the returned list has no effect on the registry.
        """
        return list(self._tasks.keys())

    def __contains__(self, run_id: object) -> bool:
        """Return ``True`` if *run_id* has a registered task (lock-free)."""
        return run_id in self._tasks

    def __len__(self) -> int:
        """Return the number of currently registered runs (lock-free)."""
        return len(self._tasks)

    def __repr__(self) -> str:
        return f"RunRegistry(active={list(self._tasks.keys())!r})"
