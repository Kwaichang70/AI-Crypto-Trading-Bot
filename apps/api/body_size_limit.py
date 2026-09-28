"""
apps/api/body_size_limit.py
----------------------------
WP1.3a round 6 (user decision): a blanket, endpoint-agnostic request-body
size cap, enforced at the ASGI transport layer -- BEFORE routing,
rate-limiting, auth, or any request-body parsing.

Why this exists
----------------
Rounds 1-5 of WP1.3a closed a whole class of DoS findings (WP13a-S-R2-01,
S-R3-02, S-R4-03, S-R5-02, S-R5-03) that all stemmed from the SAME root
cause: a single request body could carry an arbitrarily large value (a
list, a dict, a string, a huge int) into ``trading.exit_config``. Each of
those fixes bounded ONE specific code path's cost. This middleware instead
bounds the attack surface itself, independent of which field or endpoint
the oversized value eventually reaches -- a defence-in-depth backstop, not
a replacement for the per-field bounds already in ``exit_config.py``.

There are no upload endpoints anywhere in ``apps/api`` (verified by
grepping every router for ``UploadFile``/``multipart``/streaming file
responses -- none exist), so a single small, uniform cap is safe for
every endpoint that accepts a body.

Design
------
A pure ASGI middleware class (NOT ``BaseHTTPMiddleware``) so it can enforce
the limit while the body is STREAMING, one chunk at a time, without ever
buffering the whole body itself:

1. Fast path -- ``Content-Length`` present and already over the limit:
   reject with 413 immediately, before ``receive()`` is called even once
   (the body is never read at all).
2. Streaming path -- chunked requests, or requests with no
   ``Content-Length`` header: wrap ``receive()`` so every
   ``http.request`` message's ``body`` chunk is counted as it arrives.
   The moment the running total exceeds the limit, this middleware sends
   its OWN complete 413 response directly (bypassing the downstream app
   entirely) and returns ``{"type": "http.disconnect"}`` to the
   downstream app instead of the oversized chunk. NOTE: a sentinel
   exception raised from inside ``receive()`` does NOT work here --
   FastAPI's own body-parsing step wraps every exception from ``receive()``
   in a blanket ``except Exception`` and converts it to its OWN
   ``HTTPException(400, ...)`` before it could ever reach a
   ``try/except`` around ``self.app(...)`` in this middleware. ``send``
   is therefore ALSO wrapped, to silently discard whatever response the
   downstream app attempts to send after this middleware's own 413 has
   already gone out (harmless -- the downstream app has no way to know
   its response was superseded, and the real client only ever receives
   the one, correct 413).

GET/HEAD/OPTIONS requests (and non-HTTP ASGI scopes, e.g. lifespan) are
passed through completely untouched -- they never carry a body, so
wrapping ``receive()`` for them would be pure overhead with no benefit.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

__all__ = ["BodySizeLimitMiddleware"]

# Methods that never carry a meaningful request body (RFC 7231/9110) -- GET,
# HEAD and OPTIONS are passed through untouched, matching every existing
# health/metrics/list endpoint's behaviour exactly (unaffected, per spec).
_BODYLESS_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


def _too_large_response_body(max_bytes: int) -> bytes:
    import json

    return json.dumps(
        {"detail": {"code": "request_body_too_large", "max_bytes": max_bytes}}
    ).encode("utf-8")


async def _send_413(send: Send, max_bytes: int) -> None:
    body = _too_large_response_body(max_bytes)
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class BodySizeLimitMiddleware:
    """Pure-ASGI middleware capping every request body at ``max_bytes``.

    Register this LAST in ``create_app()`` (i.e. via the final
    ``application.add_middleware(...)`` call) -- Starlette makes the
    MOST RECENTLY added middleware the OUTERMOST layer, so this must sit
    outside CORS, request-timing and rate-limiting, and long before any
    dependency (auth, pydantic body parsing) ever runs.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            # Non-HTTP scopes (lifespan, websocket) -- nothing to guard.
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        if method in _BODYLESS_METHODS:
            await self.app(scope, receive, send)
            return

        # Fast path (SY-13a class DoS, transport layer): reject BEFORE
        # reading any body at all when the client declared it up front.
        for name, value in scope.get("headers") or ():
            if name.lower() == b"content-length":
                try:
                    declared_length = int(value)
                except (TypeError, ValueError):
                    declared_length = None
                if declared_length is not None and declared_length > self.max_bytes:
                    await _send_413(send, self.max_bytes)
                    return
                break

        # Streaming path: guard EVERY chunk as it arrives, for chunked
        # requests or requests with no Content-Length header at all.
        # Never buffers -- only a running integer counter is kept. See the
        # module docstring for why BOTH ``receive`` and ``send`` must be
        # wrapped (a plain exception from ``receive()`` alone would be
        # swallowed and re-mapped to a 400 by FastAPI's own body parser).
        total_received = 0
        responded = False

        async def _guarded_send(message: Message) -> None:
            if responded:
                return
            await send(message)

        async def _guarded_receive() -> Message:
            nonlocal total_received, responded
            message = await receive()
            if message.get("type") == "http.request" and not responded:
                total_received += len(message.get("body") or b"")
                if total_received > self.max_bytes:
                    responded = True
                    await _send_413(send, self.max_bytes)
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, _guarded_receive, _guarded_send)
