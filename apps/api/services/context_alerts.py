"""Telegram alerts for the live context poller (CF-24-01 P6, SY-01-13, SY-01-21j).

``ContextAlertListener`` implements ``StatusListener`` on the one ``TelegramNotifier`` (read lazily
from the container). Texts are fixed strings plus a closed series label: no value, exception or URL
text, no ``<``, ``>`` or ``&`` (HTML). Failures log ``error_type`` only; the poller's background
tasks call this, so nothing here blocks its schedule.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final

import structlog

from common.logging import exc_log_fields
from data.live_context import LIVE_CONSUMERS
from data.live_context_policy import STATUS_REMINDER_S

log = structlog.get_logger(__name__)

SEND_TIMEOUT_S: Final = 15.0  # above the notifier's own 10 s total timeout
_SERIES: Final = {  # closed: an unknown series id is never interpolated into a message
    "alternative_me:fgi": "Fear and Greed index",
    "coingecko:btc_dominance": "BTC dominance",
    "fred:T10Y2Y": "T10Y2Y yield curve",
    "whale:net_flow_1h": "Whale net flow",
}
_STALE: Final = "Live context series {} is stale or its provider is failing."
_FRESH: Final = "Live context series {} is fresh again."
_DEAD: Final = "The live context poller has stopped; each series goes neutral as it ages out."
_RECORDER: Final = "Context recording is failing; recorded history is being lost."
_DEAD_KEY: Final = "poller_dead"


class ContextAlertListener:
    """Stale/recovered messages per series, poller-dead and recorder-failing, 6 h re-notify."""

    def __init__(
        self,
        notifier: Callable[[], Any],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._notifier, self._sleep, self._clock = notifier, sleep, clock
        self._announced: dict[str, bool] = {}  # open stale condition -> a message was attempted
        self._tasks: dict[str, asyncio.Task[None]] = {}  # reminder per open condition
        self._recorder_sent: float | None = None

    # -- StatusListener ----------------------------------------------------
    async def on_status_change(self, series: str, old: str, new: str) -> None:
        if series not in _SERIES or "not_configured" in (old, new) or old == new:
            return
        name = _SERIES[series]
        if new != "fresh" and series not in self._tasks:  # opens once; QR-P6-01: boot too
            target = self._target("stale")
            self._announced[series] = target is not None
            self._remind(series)
            if target is not None:
                await self._deliver(target, _STALE.format(name))
        elif new == "fresh":
            announced = self._announced.pop(series, False)
            if (task := self._tasks.pop(series, None)) is not None:
                task.cancel()
            if announced and (target := self._target("stale")) is not None:
                await self._deliver(target, _FRESH.format(name))

    async def on_poller_dead(self) -> None:
        if _DEAD_KEY in self._tasks:
            return
        self._remind(_DEAD_KEY)
        if (target := self._target("dead")) is not None:
            await self._deliver(target, _DEAD)

    async def on_recorder_failing(self) -> None:
        now = self._clock()
        if self._recorder_sent is not None and now - self._recorder_sent < STATUS_REMINDER_S:
            return
        if (target := self._target("recorder")) is not None:
            self._recorder_sent = now
            await self._deliver(target, _RECORDER)

    async def aclose(self) -> None:
        """Cancel the reminders; sends nothing and never raises."""
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- internals ---------------------------------------------------------
    def _target(self, kind: str) -> tuple[Any, str] | None:
        """(notifier, level) when a message of ``kind`` may be sent now, else ``None``."""
        try:
            counts = LIVE_CONSUMERS.counts()
            notifier = self._notifier()
        except Exception as exc:
            log.warning("context_alerts.notifier_unavailable", **exc_log_fields(exc))
            return None
        if notifier is None or (kind == "stale" and not any(counts.values())):
            return None
        critical = kind == "dead" or (kind == "stale" and counts.get("live", 0) > 0)
        return notifier, "CRITICAL" if critical else "WARNING"

    async def _deliver(self, target: tuple[Any, str], text: str) -> None:
        notifier, level = target
        try:
            sent = await asyncio.wait_for(
                notifier.send_alert(alert_type="mtf_context", level=level, message=text),
                SEND_TIMEOUT_S,
            )
            if not sent:
                log.warning("context_alerts.send_failed", delivered=False)
        except Exception as exc:  # TimeoutError included; never reaches the poller
            log.warning("context_alerts.send_failed", **exc_log_fields(exc))

    def _remind(self, key: str) -> None:
        if key not in self._tasks:
            self._tasks[key] = asyncio.get_running_loop().create_task(self._reminder(key))

    async def _reminder(self, key: str) -> None:
        while True:
            await self._sleep(STATUS_REMINDER_S)
            dead = key == _DEAD_KEY
            if (target := self._target("dead" if dead else "stale")) is None:
                continue
            if not dead:
                self._announced[key] = True
            text = _DEAD if dead else _STALE.format(_SERIES[key])
            await self._deliver(target, text)
