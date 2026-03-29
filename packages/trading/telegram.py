"""
packages/trading/telegram.py
------------------------------
Telegram Bot API integration for trading alerts.

Usage::

    client = TelegramNotifier(bot_token="123:ABC", chat_id="-100123")
    await client.send_alert(alert_type="circuit_breaker", level="CRITICAL", message="Halt.")
    await client.send_trade(symbol="BTC/USDT", side="BUY", quantity="0.01", price="50000")
    await client.close()

All send methods are fire-and-forget safe: they catch every exception and
return ``False`` on failure rather than raising.  The caller should never
``await`` on these for error handling.  Use ``asyncio.create_task`` to avoid
blocking the engine loop.

The HTTP session is lazily created on the first call and reused for all
subsequent requests.  Call ``close()`` in the application shutdown sequence
to drain the underlying TCP connection pool.
"""

from __future__ import annotations

import aiohttp
import structlog

__all__ = ["TelegramNotifier"]

logger = structlog.get_logger(__name__)


class TelegramNotifier:
    """Send trading alerts and trade notifications to Telegram.

    Parameters
    ----------
    bot_token:
        Telegram Bot API token obtained from @BotFather.
        Format: ``<numeric_id>:<alphanumeric_string>``.
    chat_id:
        Target chat or group identifier.
        Positive integers for direct chats; negative for groups/channels
        (e.g. ``"-100123456789"``).
    """

    _BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._session: aiohttp.ClientSession | None = None
        self._url = self._BASE_URL.format(token=bot_token)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (or lazily create) the shared aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # ------------------------------------------------------------------
    # Public send API
    # ------------------------------------------------------------------

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send an arbitrary message to the configured Telegram chat.

        Parameters
        ----------
        text:
            Message body.  HTML tags are accepted when ``parse_mode="HTML"``.
        parse_mode:
            Telegram parse mode.  ``"HTML"`` or ``"Markdown"``.  Default
            ``"HTML"`` because it is more lenient with special characters.

        Returns
        -------
        bool
            ``True`` on HTTP 200, ``False`` on any error (network / API).
        """
        try:
            session = await self._get_session()
            async with session.post(
                self._url,
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return True
                body = await resp.text()
                logger.warning(
                    "telegram.send_failed",
                    status=resp.status,
                    body=body[:200],
                )
                return False
        except Exception as exc:
            logger.warning("telegram.send_error", error=str(exc)[:200])
            return False

    async def send_trade(
        self,
        symbol: str,
        side: str,
        quantity: str,
        price: str,
        pnl: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        """Format and send a trade execution notification.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTC/USDT"``.
        side:
            ``"BUY"`` or ``"SELL"`` (case-insensitive).
        quantity:
            Executed quantity as a string.
        price:
            Execution price as a string.
        pnl:
            Realised PnL string, e.g. ``"12.50"``.  ``None`` for open trades.
        run_id:
            Run UUID string.  First 8 characters shown as a short reference.

        Returns
        -------
        bool
            Forwarded from :meth:`send_message`.
        """
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        pnl_line = ""
        if pnl is not None:
            try:
                pnl_float = float(pnl)
                pnl_emoji = "💰" if pnl_float >= 0 else "📉"
                pnl_line = f"\n{pnl_emoji} PnL: <b>{pnl}</b>"
            except ValueError:
                pnl_line = f"\nPnL: {pnl}"

        text = (
            f"{emoji} <b>{side.upper()}</b> {symbol}\n"
            f"Qty: {quantity} @ {price}"
            f"{pnl_line}"
        )
        if run_id:
            text += f"\nRun: <code>{run_id[:8]}</code>"

        return await self.send_message(text)

    async def send_alert(
        self,
        alert_type: str,
        level: str,
        message: str,
        run_id: str | None = None,
    ) -> bool:
        """Format and send an alert notification.

        Parameters
        ----------
        alert_type:
            Machine-readable alert type, e.g. ``"circuit_breaker"``.
            Controls the type-specific emoji prefix.
        level:
            Severity string: ``"INFO"``, ``"WARNING"``, or ``"CRITICAL"``.
        message:
            Human-readable alert description.
        run_id:
            Optional run UUID string.

        Returns
        -------
        bool
            Forwarded from :meth:`send_message`.
        """
        level_emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "CRITICAL": "🚨",
        }.get(level.upper(), "📢")

        type_emoji = {
            "circuit_breaker": "🛑",
            "rollback": "↩️",
            "learning_disabled": "🚫",
            "regime_change": "🔄",
            "ath_equity": "🏆",
            "equity_warning": "📉",
            "daily_loss_limit": "⛔",
            "trade_milestone": "🎯",
            "model_retrained": "🤖",
        }.get(alert_type.lower(), "📢")

        text = (
            f"{level_emoji}{type_emoji} <b>{alert_type.upper()}</b>\n"
            f"{message}"
        )
        if run_id:
            text += f"\nRun: <code>{run_id[:8]}</code>"

        return await self.send_message(text)

    async def send_daily_summary(
        self,
        equity: str,
        daily_pnl: str,
        total_trades: int,
        winning: int,
        losing: int,
        drawdown_pct: float,
    ) -> bool:
        """Send a daily portfolio summary digest.

        Parameters
        ----------
        equity:
            Current total equity as a formatted string (e.g. ``"$10,250.00"``).
        daily_pnl:
            Net PnL for the day (positive or negative) as a string.
        total_trades:
            Total trades executed today.
        winning:
            Number of winning trades.
        losing:
            Number of losing trades.
        drawdown_pct:
            Maximum intraday drawdown as a decimal fraction (e.g. ``0.032``).

        Returns
        -------
        bool
            Forwarded from :meth:`send_message`.
        """
        try:
            pnl_float = float(daily_pnl)
            pnl_emoji = "📈" if pnl_float >= 0 else "📉"
        except ValueError:
            pnl_emoji = "📊"

        text = (
            f"📋 <b>Daily Summary</b>\n"
            f"💰 Equity: <b>{equity}</b>\n"
            f"{pnl_emoji} Daily PnL: <b>{daily_pnl}</b>\n"
            f"📊 Trades: {total_trades} ({winning}W/{losing}L)\n"
            f"📉 Max Drawdown: {drawdown_pct:.1%}"
        )
        return await self.send_message(text)

    async def close(self) -> None:
        """Close the underlying HTTP session and release connections.

        Should be called once during application shutdown.  Idempotent --
        safe to call multiple times or when the session was never opened.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
