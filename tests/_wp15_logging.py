"""
tests/_wp15_logging.py
------------------------
Autouse fixture for the WP1.5a tests: make structlog pass EVERY level to
``structlog.testing.capture_logs`` regardless of what an earlier test (or the
app lifespan) configured globally.

``capture_logs`` only swaps processors; a level-filtering ``wrapper_class``
(``make_filtering_bound_logger(INFO)``) drops debug events before any processor
runs, which made event-based assertions order-dependent across the full suite.
Import the fixture into a test module with::

    from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog


@pytest.fixture(autouse=True)
def _structlog_unfiltered() -> Iterator[None]:
    previous = structlog.get_config()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(0),
        cache_logger_on_first_use=False,
    )
    try:
        yield
    finally:
        structlog.configure(**previous)
