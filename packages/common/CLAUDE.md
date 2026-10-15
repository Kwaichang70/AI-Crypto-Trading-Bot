IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## Common Package
Shared types, utilities, and configuration used across all packages.

### Expected Contents
- Shared Pydantic models / type definitions
- Configuration loading and validation
- Utility functions (timestamp helpers, etc.)
- Constants and enums (OrderType, Signal, RunMode, etc.)
- Logging configuration
- Run ID generation and tracking

### Log redaction (CF-36 layer 1) -- rules for this folder
- `redaction.py` is the last stop before the renderer on BOTH log paths: the structlog chain (`config.configure_structlog`) and the stdlib bridge (`logging._configure_stdlib_bridge`). The order is `format_exc_info` -> `redact_event_dict` -> `RedactingRenderer(renderer)` on both. Never reorder, never insert a text-producing processor after `redact_event_dict`, never switch the console renderer to `rich`/`show_locals` (`build_renderer` pins `plain_traceback`).
- `redaction.py` must never call a logger, never store or echo exception text from its own failures, and imports stdlib only at module level (`common.metrics` is loaded lazily; do not use `import common.metrics as m`, the package re-exports the `metrics` singleton under that name).
- Every unbounded regex quantifier in a rule must follow a literal anchor that cannot repeat inside the quantified class (ReDoS rule, ARB-CF36-03). A new rule needs a near-miss fragment in `tests/unit/test_cf36_redaction.py::_FRAGMENTS` and a gate in `_Rule`.
- R-08 / R-10 / R-11 share ONE credential-name grammar (`_CRED_NAMES`); add names only there. A bare `key` is text-only (R-08), never a dict-key rule. Also: no rule may start inside a run of its own leading class (negative lookbehind or `_NAME_START`), replacements are precompiled (`_compile_template`, never `Match.expand`), and the T-18 corpus must include dense-match fragments.
- `configure_logging` leaves `uvicorn.access` alone when uvicorn runs with `--no-access-log` (no handlers, `propagate=False`): setting `propagate=True` there silently re-enables the access log. When the access log is on, `_AccessQueryFilter` strips the query string.
- `_WsQueryFilter` on `uvicorn.error` strips the query string from uvicorn's WebSocket handshake lines (`'%s - "WebSocket %s" [accepted]|403|%d'`). uvicorn logs them for EVERY Upgrade request, also on paths without a WebSocket route and also under `--no-access-log` (CF-A, CF36A-SEC-204). Run uvicorn at its default log level: with `--log-level debug`/`trace` the `websockets` library logs the raw request line and headers on `uvicorn.error`, which no filter covers.
- The walker is identity-preserving: a value is replaced only when redaction changed its text. Do not convert `set`/`bytes`/objects to other types; normal logs must stay byte-identical (`test_t15_*`).
- Registered secrets live in the opaque `_SecretMatcher`. Do not add accessors for the values or patterns. Values shorter than 8 characters, trivial values and short all-digit values are never registered.
- New `Settings` secret fields must be `SecretStr` (auto-registered by `api.config.register_settings_secrets`) or be added to `_PLAIN_SECRET_FIELDS`; `tests/unit/test_cf36_settings.py::test_t19_name_meta_test_*` fails otherwise.
- Never log `str(exc)`, `repr(exc)` or `exc_info` at outbound-I/O sites (ccxt/aiohttp text embeds the signed URL, `api_key=`, the Telegram token). Use `**exc_log_fields(exc)`; layer 1 is the backstop, not the control.
- Site convention (CF-36 layer 2, cf36b): at an outbound-I/O `except`, log `**exc_log_fields(exc)` (`error_type`, `http_status`, `error_at`) plus safe numeric context. Where a traceback helps diagnosis, add a SEPARATE debug event `"<event>.trace"` with `exc_info=exc` right after it (INFO drops it before any processor; layer 1 redacts it at DEBUG). Never `.exception()` / `exc_info=True` at an outbound-I/O site at INFO.
- Exchange payload dicts (tickers, orders, `info`) are never logged whole: use `bounded_scalar(v)` per selected field plus `has_info="info" in payload`.
- A domain error wrapping a ccxt/aiohttp exception uses fixed text plus `({type(exc).__name__})` and `raise ... from None`; never `{exc}` in the message (it is the `detail` of HTTP responses and the text of every downstream `error=` field). Log the original with `market_data.raise.trace` (debug) first.
- Operator rule (L-19): a live image runs with `LOG_LEVEL` other than `DEBUG`; the `.trace` events carry tracebacks.
- Layer-1 tests must use `tests/_cf36_logging.production_logging` (real `configure_logging`); `structlog.testing.capture_logs` replaces the processors and proves nothing about layer 1.
- Deploy check: `python -m api.log_hygiene_check` (prints `CF36 log hygiene: PASS ...`, exit 0).
