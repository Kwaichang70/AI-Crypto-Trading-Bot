"""
packages/common/redaction.py
----------------------------
Central log redaction (CF-36 layer 1, G-LIVE L-19).

This module is the last line of defence between application text and the log
sinks.  It is wired into BOTH output paths of the process:

* the structlog chain (``common.config.configure_structlog``), and
* the stdlib bridge (``common.logging._configure_stdlib_bridge``),

in the same order on both: ``format_exc_info`` -> :func:`redact_event_dict` ->
:class:`RedactingRenderer` (renderer wrapped with an exact-match final pass).

Design rules (binding, see ``reports/vp2-cf36/synthesis-spec.md``)
------------------------------------------------------------------
* Imports are stdlib-only at module level.  ``common.metrics`` and ``structlog``
  are imported lazily inside ``try`` blocks.  The module NEVER calls a logger.
* Fail-safe: nothing in here may raise into a log call.  Only ``Exception`` is
  caught, never ``BaseException``.  Failure text is never stored or echoed.
* ReDoS rules (spec 5.2, amended by CF36A-fix): (1) every unbounded quantifier
  follows a literal anchor that cannot repeat inside the quantified class;
  (2) no rule may START inside a run of its own leading class (negative
  lookbehind on that class, or ``_NAME_START``); (3) value classes that can
  overlap the next match are possessive or bounded; (4) replacements are
  precompiled (``_compile_template``), never ``Match.expand``; (5) the T-18
  corpus includes dense-match fragments.  Input is size-capped.
* R-08 / R-10 / R-11 share ONE credential-name grammar (``_CRED_NAMES``): add
  names only there.  A bare ``key`` is text-only (R-08), never a dict-key rule.
* The walker is identity-preserving: a value is replaced only when redaction
  changed its text, so ordinary logs are byte-identical to the unredacted chain.
* Registered secrets live in an opaque :class:`_SecretMatcher`; there is no
  public accessor for the values or the compiled patterns.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NoReturn, SupportsIndex
from urllib.parse import quote, quote_plus, unquote, urlsplit

if TYPE_CHECKING:
    from common.metrics import MetricsCollector

__all__ = [
    "REDACTED",
    "REDACTION_FAILED",
    "RedactingRenderer",
    "redact_event_dict",
    "redact_text",
    "register_secret",
    "register_secrets_from_env",
    "registered_secret_count",
    "secret_skip_reason",
    "secret_variants",
]

REDACTED = "[REDACTED]"
REDACTION_FAILED = "[REDACTION_FAILED]"

#: Strings longer than this are redacted over ``LIMIT + MARGIN`` chars and cut.
LIMIT = 65_536
MARGIN = 4_096
#: Keys whose values keep head AND tail when over ``LIMIT`` (tracebacks).
HEAD_TAIL_KEYS = frozenset({"exception", "stack", "stack_info"})
#: Keys produced by processors: only the exact-match registry is applied.
PROCESSOR_KEYS = frozenset({"timestamp", "level", "logger", "service", "env"})
MIN_SECRET_LEN = 8
MIN_DIGIT_SECRET_LEN = 12
TRIVIAL = frozenset(
    {"true", "false", "none", "null", "changeme", "password", "secret", "replace_me"}
)

_MAX_DEPTH = 6
_MAX_ITEMS = 200
_LEVEL_NAMES = frozenset({"debug", "info", "warning", "error", "critical", "exception"})

_FAILED_LINE = '{"event": "[REDACTION_FAILED]", "level": "error", "redaction_error": true}'

# ---------------------------------------------------------------------------
# Metrics hooks (lazy, never raise, never log)
# ---------------------------------------------------------------------------

_metrics_mod: Any = None


def _metrics() -> MetricsCollector:
    global _metrics_mod
    if _metrics_mod is None:
        # NOT ``import common.metrics as m``: ``common/__init__`` re-exports the
        # ``metrics`` singleton under the same name, which shadows the submodule
        # attribute on the package.
        import importlib

        _metrics_mod = importlib.import_module("common.metrics")
    collector: MetricsCollector = _metrics_mod.metrics
    return collector


def _count_hit(rule_id: str, n: int = 1) -> None:
    with contextlib.suppress(Exception):
        _metrics().increment("log_redaction_hits_total", n, labels={"rule": rule_id})


def _count_failure() -> None:
    with contextlib.suppress(Exception):
        _metrics().increment("log_redaction_failures_total")


# ---------------------------------------------------------------------------
# Rule set (compiled once at import)
# ---------------------------------------------------------------------------


_TEMPLATE_REF = re.compile(r"\\(\d)")


def _compile_template(template: str) -> Callable[[re.Match[str]], str]:
    """Pre-split ``\\N`` templates once; per match only group lookups and a join.

    ``Match.expand`` re-parses the template on every match (~10 us); dense-match
    inputs (``key=a&key=a&...``) then dominate the near-miss budget.  Unmatched
    groups render as "" (as ``re`` does since 3.5).
    """
    parts = _TEMPLATE_REF.split(template)
    lits, refs = parts[0::2], [int(g) for g in parts[1::2]]
    if not refs:

        def const(_m: re.Match[str]) -> str:
            return template

        return const

    def repl(m: re.Match[str]) -> str:
        out = [lits[0]]
        for i, g in enumerate(refs):
            out.append(m.group(g) or "")
            out.append(lits[i + 1])
        return "".join(out)

    return repl


class _Rule:
    """One redaction rule.

    ``gate`` is a tuple of any-of groups of lowercase substrings: the rule is
    only attempted when EVERY group has at least one member in the (lowered)
    text.  Each gate is a necessary condition of the rule's pattern, so gating
    never changes the output (pinned by the gate-soundness test); it only skips
    regex scans that cannot match.  The replacement is a precompiled callable
    (:func:`_compile_template`), never ``Match.expand``.

    Gate needles are matched on ``str.lower()``; for non-ASCII case folding
    (e.g. U+017F) the gated path can differ from ``re.I``.  This is accepted
    (CF36A-CR-07).
    """

    __slots__ = ("gate", "id", "repl", "rx")

    def __init__(
        self,
        rule_id: str,
        pattern: str,
        template: str,
        gate: tuple[tuple[str, ...], ...],
        flags: int = 0,
    ) -> None:
        self.id = rule_id
        self.rx = re.compile(pattern, flags)
        self.gate = gate
        self.repl: Any = _compile_template(template)

    def applies(self, lowered: str) -> bool:
        for group in self.gate:
            for needle in group:
                if needle in lowered:
                    break
            else:
                return False
        return True


#: Shared credential-NAME grammar (lowercase; separators optional).  Single source
#: for R-08 (``name=``), R-11 (``name:`` / quoted keys) and R-10 (dict keys), so the
#: native and stdlib paths agree by construction (CF36A-SEC-03).  Add names ONLY here.
_CRED_NAMES = (
    r"api[_-]?key|apikey|api[_-]?secret|api[_-]?sign|api[_-]?passphrase"
    r"|(?:secret|access|private|signing|admin|master|encryption|hmac|session)[_-]?key"
    r"|access[_-]?key[_-]?id|access[_-]?secret|client[_-]?secret"
    r"|(?:access|refresh|auth|bot|bearer|session|id|confirm|csrf)[_-]?token"
    r"|secret|token|pass[_-]?word|passwd|pwd|pass[_-]?phrase|signature|sign|sig"
    r"|authorization|auth|cookie|dsn|credentials?"
)
#: A name starts after a non-alphanumeric (incl. "_", "-", "."), or at a camelCase
#: boundary.  The (?-i:...) groups keep the case test case-sensitive under re.I.
_NAME_START = r"(?:(?<![A-Za-z0-9])|(?-i:(?<=[a-z0-9])(?=[A-Z]))|(?-i:(?<=[A-Z])(?=[A-Z][a-z])))"
#: Necessary-condition needles of every _CRED_NAMES alternative (and bare "key").
_NAME_NEEDLES = (
    "key",
    "secret",
    "token",
    "pass",
    "pwd",
    "sig",
    "auth",
    "cookie",
    "dsn",
    "credential",
)
#: Quoted value: possessive up to the closing quote or end of line (no length bound,
#: so no plaintext tail survives: CF36A-CR-201); an existing [REDACTED...] is never
#: re-matched.
_QVAL = r"""([bB]?['"])(?!\[REDACTED)[^'"\r\n]++"""

_RULES: tuple[_Rule, ...] = (
    # R-01 PEM private key (to the END marker or end of string).
    _Rule(
        "R-01",
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
        "[REDACTED_PRIVATE_KEY]",
        (("-----begin",),),
        re.S,
    ),
    # R-02 URL userinfo password.  Scheme bounded to {0,30} (ARB-CF36-03: the
    # unbounded form was super-linear on dotted runs).
    _Rule(
        "R-02",
        r"""\b([a-z][a-z0-9+.\-]{0,30}://)([^:/?#@\s'"]{0,256}):([^@/\s'"]+)@""",
        r"\1\2:[REDACTED]@",
        (("://",),),
        re.I,
    ),
    # R-03 Telegram token in a URL path.
    _Rule(
        "R-03",
        r"\bbot(\d{5,}):[A-Za-z0-9_-]{20,}",
        r"bot\1:[REDACTED]",
        (("bot",), (":",)),
        re.I,
    ),
    # R-04 bare Telegram token.
    _Rule("R-04", r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b", "[REDACTED_TG_TOKEN]", ((":",),)),
    # R-05 credential headers (dict reprs, "Header: value").  The value class
    # excludes "[" so an existing [REDACTED] is never re-matched.
    _Rule(
        "R-05",
        r"""\b(authorization|proxy-authorization|x-mbx-apikey|cb-access-(?:key|sign|passphrase)"""
        r"""|ok-access-(?:key|sign|passphrase)|x-bapi-(?:api-key|sign)"""
        r"""|kc-api-(?:key|sign|passphrase)|api-key|api-sign|x-api-key|x-admin-key"""
        r"""|x-live-confirm-token)(['"]?\s*[:=]\s*['"]?)(?:(?:bearer|basic|token)\s+)?[^\s'",}\[]+""",
        r"\1\2[REDACTED]",
        # every header name above contains one of the second group's members
        ((":", "="), ("authoriz", "-key", "-sign", "-passphrase", "-token", "apikey")),
        re.I,
    ),
    # R-06 free-text bearer token.
    _Rule(
        "R-06",
        r"\bbearer\s+[A-Za-z0-9._~+/=-]{8,}",
        "Bearer [REDACTED]",
        (("bearer",),),
        re.I,
    ),
    # R-07 JWT.  Lookbehind on the rule's own leading class: at most one start per
    # [A-Za-z0-9_-] run, so the scan is linear (CF36A-SEC-02).  Possessive segments.
    _Rule(
        "R-07",
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{5,}+\.[A-Za-z0-9_-]{5,}+\.[A-Za-z0-9_-]{5,}+",
        "[REDACTED_JWT]",
        (("eyj",),),
    ),
    # R-08 query / form / key=value parameters (value bare or quoted).  A bare
    # None / null / boolean that is the WHOLE value is kept (CF36A-SEC-202).
    _Rule(
        "R-08",
        r"(" + _NAME_START + r"(?:" + _CRED_NAMES + r")|(?<![A-Za-z0-9_])key)"
        r"""(=|%3D)(?:"""
        + _QVAL
        + r"""|(?!(?:none|null|true|false)(?:[&\s'"<>,;#)\]}\[]|\Z))[^&\s'"<>,;#)\]}\[]+)""",
        r"\1\2\3[REDACTED]",
        (("=", "%3"), _NAME_NEEDLES),
        re.I,
    ),
    # R-11 ``name: value`` / quoted keys (JSON, dict repr, ``api_key: v``).  Skips
    # None / null / booleans only as the WHOLE value (CF36A-SEC-202), mirroring R-10.
    _Rule(
        "R-11",
        r"(" + _NAME_START + r"(?:" + _CRED_NAMES + r"))"
        r"""(['"]?[ \t]{0,4}:[ \t]{0,4})(?:"""
        + _QVAL
        + r"""|(?!(?:none|null|true|false)(?:[\s'",}\]\[()<>]|\Z))[^\s'",}\]\[()<>]+)""",
        r"\1\2\3[REDACTED]",
        ((":",), _NAME_NEEDLES),
        re.I,
    ),
)

#: Reference form of the cheap guard (spec §5.2, ARB-CF36-09).  Kept for the
#: T-36 completeness meta-test only; the hot path uses :func:`_prefilter_hit`,
#: which is far cheaper (``re.I`` alternation costs ~15 us per 100 chars).
_PREFILTER = re.compile(
    r"(=|:|%3d|://|bot\d|bearer|authoriz|eyj|-----begin|apikey|api-key|access-|-sign"
    r"|passphrase|admin-key|confirm-token|\d{6,12}:)",
    re.I,
)


def _prefilter_hit(lowered: str) -> bool:
    """True when ``lowered`` (already ``.lower()``) may contain a credential.

    Sound reduction of ``_PREFILTER``: every rule R-01..R-08 can only match a
    text that contains at least one of ``=``, ``:``, ``%3`` (R-08 ``%3D``),
    ``bearer`` (R-06), ``eyj`` (R-07) or ``-----begin`` (R-01).  R-02..R-05
    and R-08 require ``:`` or ``=``; the remaining spec members (``apikey``,
    ``access-``, ``-sign``, ``passphrase`` ...) only occur in rule matches that
    also contain ``:`` or ``=``.  C-level substring tests; ~0.3 us regardless
    of length.  Skipping rules that cannot match never changes the output
    (pinned by the T-36 equivalence test).
    """
    return (
        "=" in lowered
        or ":" in lowered
        or "%3" in lowered
        or "bearer" in lowered
        or "eyj" in lowered
        or "-----begin" in lowered
    )


#: R-10: sensitive dict KEY names (value is redacted whole).  The key is normalised
#: (camelCase -> snake_case, "." -> "_", lowercase) and matched suffix-anchored, so
#: the normaliser keeps the LAST 128 characters.
_SENSITIVE_KEY = re.compile(r"(?:^|[_-])(?:" + _CRED_NAMES + r")\Z")
_CAMEL_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")

_SAFE_PROCESSOR_TEXT = re.compile(r"^[A-Za-z0-9_.:+\-]{1,200}\Z")
_PLAIN_EVENT = re.compile(r"^[a-z0-9_.:-]{1,120}\Z")
_PLAIN_LOGGER = re.compile(r"^[A-Za-z0-9_.]{1,200}\Z")


def _norm_key(key: str) -> str:
    return _CAMEL_2.sub("_", _CAMEL_1.sub("_", key[-128:])).replace(".", "_").lower()


@lru_cache(maxsize=4096)
def is_sensitive_key(key: str) -> bool:
    """True when ``key`` names a credential (R-10).  Cached; suffix-anchored."""
    return _SENSITIVE_KEY.search(_norm_key(key)) is not None


# ---------------------------------------------------------------------------
# Exact-match registry (R-09), opaque (ARB-CF36-11)
# ---------------------------------------------------------------------------


class _SecretMatcher:
    """Immutable, opaque exact-match engine.

    Neither ``repr`` nor ``str`` nor pickling reveals the values or the
    compiled patterns (``re.Pattern.__repr__`` prints the pattern).
    """

    __slots__ = ("_rx_all", "_rx_final", "_values")

    def __init__(self, values: frozenset[str]) -> None:
        ordered = sorted(values, key=lambda s: (-len(s), s))
        self._values = values
        self._rx_all = re.compile("|".join(re.escape(v) for v in ordered)) if ordered else None
        # ARB-CF36-12: the final pass runs on rendered JSON; a variant holding
        # a quote or backslash could match across a string boundary and break
        # the line.  The walker-level pass already covers those secrets.
        final = [v for v in ordered if '"' not in v and "\\" not in v]
        self._rx_final = re.compile("|".join(re.escape(v) for v in final)) if final else None

    def subn_all(self, s: str) -> tuple[str, int]:
        rx = self._rx_all
        if rx is None:
            return s, 0
        return rx.subn(REDACTED, s)

    def sub_final(self, s: str) -> str:
        rx = self._rx_final
        if rx is None:
            return s
        return rx.sub(REDACTED, s)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return "<SecretMatcher>"

    __str__ = __repr__

    def __reduce__(self) -> NoReturn:
        raise TypeError("_SecretMatcher cannot be pickled")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("_SecretMatcher cannot be pickled")

    def __getstate__(self) -> NoReturn:
        raise TypeError("_SecretMatcher cannot be pickled")


_REG_LOCK = threading.Lock()
_MATCHER: _SecretMatcher = _SecretMatcher(frozenset())


def _reset_registry_for_tests() -> None:
    """Drop every registered secret (test isolation only)."""
    global _MATCHER
    with _REG_LOCK:
        _MATCHER = _SecretMatcher(frozenset())


def _get_registry() -> _SecretMatcher:
    """Return the current (opaque) matcher, for snapshot/restore in tests and the self-check."""
    return _MATCHER


def _set_registry(matcher: _SecretMatcher) -> None:
    global _MATCHER
    with _REG_LOCK:
        _MATCHER = matcher


def registered_secret_count() -> int:
    """Number of registered exact-match variants.  Never the values."""
    return len(_MATCHER)


def _variant_ok(x: str) -> bool:
    if len(x) < MIN_SECRET_LEN or x.isspace():
        return False
    if x.isdigit() and len(x) < MIN_DIGIT_SECRET_LEN:
        return False
    return x.lower() not in TRIVIAL


def secret_skip_reason(value: str | None, kind: str = "plain") -> str | None:
    """Why ``value`` would not be registered, or ``None`` when it is registrable.

    Returns one of ``too_short``, ``trivial``, ``digits_only`` (never the value
    or its length).  Empty / whitespace-only / ``None`` values and URL-kind
    values without a password return ``None`` (nothing to protect, no warning).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    if secret_variants(value, kind):
        return None
    base = _url_password(value) if kind == "url" else value
    if not base or not base.strip():
        return None
    probe = base.strip()
    if len(probe) < MIN_SECRET_LEN:
        return "too_short"
    if probe.lower() in TRIVIAL:
        return "trivial"
    if probe.isdigit() and len(probe) < MIN_DIGIT_SECRET_LEN:
        return "digits_only"
    return "trivial"


_PCT_HEX = re.compile(r"%[0-9A-F]{2}")


def _text_variants(v: str) -> set[str]:
    out = {
        v,
        quote(v, safe=""),
        quote_plus(v),
        json.dumps(v)[1:-1],
        json.dumps(v, ensure_ascii=False)[1:-1],
        v.encode("unicode_escape").decode("ascii"),
        quote(v),  # urllib default, safe="/"
        quote_plus(v, safe="!$'()*,/:?@"),  # yarl/aiohttp query value (verified, yarl 1.22.0)
        quote(v, safe="/:@!$'()*,;=+&"),  # yarl path segment (verified)
    }
    out |= {_PCT_HEX.sub(lambda m: m.group(0).lower(), x) for x in list(out) if "%" in x}
    stripped = v.strip()
    if stripped and stripped != v:
        out.add(stripped)
    return out


def _url_password(value: str) -> str | None:
    try:
        return urlsplit(value).password
    except ValueError:
        return None


_TG_TOKEN = re.compile(r"^\d+:[\w-]+$")


def pem_body(value: str) -> str:
    """Header-stripped, joined PEM body (equals ``run_orchestrator._normalize_exchange_secret``)."""
    normalized = value.replace("\\n", "\n")
    lines = [
        line.strip() for line in normalized.splitlines() if line.strip() and "-----" not in line
    ]
    if lines:
        return "".join(lines)
    return normalized


def _expand_impl(value: str, kind: str) -> set[str]:
    bases: set[str] = set()
    if kind == "url":
        pw = _url_password(value)
        if pw:
            bases.add(pw)
            bases.add(unquote(pw))
    elif kind == "telegram":
        bases.add(value)
        if _TG_TOKEN.match(value):
            bases.add(value.split(":", 1)[1])
    elif kind == "pem":
        bases.add(value)
        bases.add(pem_body(value))
        for line in value.replace("\\n", "\n").splitlines():
            line = line.strip()
            if len(line) >= 16 and "-----" not in line:
                bases.add(line)
    else:
        bases.add(value)
    out: set[str] = set()
    for b in bases:
        if b and b.strip() and _variant_ok(b.strip()):
            out |= {x for x in _text_variants(b) if _variant_ok(x)}
    return out


def secret_variants(value: str, kind: str = "plain") -> frozenset[str]:
    """Exact-match variants :func:`register_secret` would add for ``value``.

    Pure function of its argument (reads no registry state); used by the
    deploy self-check to know what must never appear in a log line.  Callers
    must never log the result.
    """
    try:
        return frozenset(_expand_impl(value, kind))
    except Exception:
        return frozenset()


def register_secret(value: str | None, *, name: str, kind: str = "plain") -> int:
    """Register ``value`` for exact-match redaction; return the variants added.

    ``kind``: ``plain`` | ``url`` (password component only) | ``telegram`` |
    ``pem``.  ``None``, empty, whitespace-only, <8-char, trivial and short
    all-digit values add nothing.  Never logs, never raises on bad input.
    """
    del name  # accepted for call-site readability / future audit; never stored
    global _MATCHER
    try:
        if not isinstance(value, str) or not value.strip():
            return 0
        variants = _expand_impl(value, kind)
        if not variants:
            return 0
        with _REG_LOCK:
            current = _MATCHER._values
            new = variants - current
            if not new:
                return 0
            _MATCHER = _SecretMatcher(current | frozenset(new))
            return len(new)
    except Exception:
        return 0


_ENV_NAME = re.compile(
    r"(^|_)(API_KEY|APIKEY|API_SECRET|SECRET|SECRET_KEY|TOKEN|BOT_TOKEN|PASSWORD|PASSPHRASE"
    r"|PRIVATE_KEY|DSN|DATABASE_URL|REDIS_URL)$",
    re.I,
)


def register_secrets_from_env(environ: Mapping[str, str] | None = None) -> int:
    """Register secrets found in the process environment (fallback source).

    Only suffix-anchored credential names are considered (ARB-CF36-14).
    Returns the number of variants added.  Never logs.
    """
    env = os.environ if environ is None else environ
    added = 0
    try:
        for name, value in list(env.items()):
            if not isinstance(name, str) or not isinstance(value, str):
                continue
            if not _ENV_NAME.search(name):
                continue
            upper = name.upper()
            if upper.endswith(("DATABASE_URL", "REDIS_URL")):
                kind = "url"
            elif upper.endswith("BOT_TOKEN"):
                kind = "telegram"
            elif "-----BEGIN" in value:
                kind = "pem"
            else:
                kind = "plain"
            added += register_secret(value, name=name, kind=kind)
    except Exception:
        return added
    return added


# ---------------------------------------------------------------------------
# Text redaction
# ---------------------------------------------------------------------------


def _core(s: str) -> str:
    m = _MATCHER  # single read; the registry swap is atomic
    s, k = m.subn_all(s)  # R-09 first (ARB-CF36-08)
    if k:
        _count_hit("R-09", k)
    lowered = s.lower()
    if _prefilter_hit(lowered):
        # ``lowered`` is taken once: replacements only remove text or add
        # ``[REDACTED...]`` markers, which hold no gate needle that could newly
        # enable a later rule, so a stale haystack is conservative.
        for rule in _RULES:
            if not rule.applies(lowered):
                continue
            s, n = rule.rx.subn(rule.repl, s)
            if n:
                _count_hit(rule.id, n)
    return s


def redact_text(s: str, *, head_tail: bool = False) -> str:
    """Redact credentials in ``s``.  Never raises; ``[REDACTION_FAILED]`` on error.

    Strings over :data:`LIMIT` are redacted over ``LIMIT + MARGIN`` characters
    and then cut; with ``head_tail`` (tracebacks) the head and the tail are
    kept, each ``LIMIT // 2``, so the final exception line survives.
    """
    try:
        if not isinstance(s, str):
            s = str(s)
        n = len(s)
        if n <= LIMIT:
            return _core(s)
        if head_tail:
            h = LIMIT // 2
            # The cut is computed AFTER shrinking replacements: text that lay beyond
            # the window edge must never be pulled into the kept part (CF36A-SEC-05).
            hw = s[: h + MARGIN]
            hr = _core(hw)
            head = hr[: max(0, h - max(0, len(hw) - len(hr)))]
            tw = s[-(h + MARGIN) :]
            tr = _core(tw)
            keep = h - max(0, len(tw) - len(tr))
            tail = tr[-keep:] if keep > 0 else ""  # never tr[-0:] (the whole string)
            return head + f"\n...[TRUNCATED len={n}]...\n" + tail
        win = s[: LIMIT + MARGIN]
        red = _core(win)
        return red[: max(0, LIMIT - max(0, len(win) - len(red)))] + f"...[TRUNCATED len={n}]"
    except Exception:
        _count_failure()
        return REDACTION_FAILED


# ---------------------------------------------------------------------------
# Walker (identity-preserving, ARB-CF36-04)
# ---------------------------------------------------------------------------


def _opaque(v: object, depth: int, text: str | None = None) -> object:
    """Redact the textual form a renderer would print; keep ``v`` if unchanged."""
    if text is None:
        sv = getattr(v, "__structlog__", None)
        if sv is not None and callable(sv):
            res = sv()
            if not isinstance(res, str):
                out = _rv(res, depth + 1)
                return v if out is res else out
            text = res
        else:
            text = repr(v)
    red = redact_text(text)
    return v if red == text else red


def _rv(v: object, depth: int, head_tail: bool = False) -> object:
    t = type(v)
    if v is None or t is bool or t is int or t is float:
        return v
    if isinstance(v, str):
        red = redact_text(v, head_tail=head_tail)
        return v if red == v else red
    if t is dict and isinstance(v, dict):
        if depth > _MAX_DEPTH or len(v) > _MAX_ITEMS:
            return _opaque(v, depth, repr(v))
        new: dict[Any, Any] | None = None
        for k, val in v.items():
            if (
                isinstance(k, str)
                and val is not None
                and type(val) is not bool
                and (is_sensitive_key(k))
            ):
                nv: Any = REDACTED
                _count_hit("R-10")
            else:
                nv = _rv(val, depth + 1)
            if nv is not val:
                if new is None:
                    new = dict(v)
                new[k] = nv
        return v if new is None else new
    if (t is list or t is tuple) and isinstance(v, (list, tuple)):
        if depth > _MAX_DEPTH or len(v) > _MAX_ITEMS:
            return _opaque(v, depth, repr(v))
        items: list[Any] | None = None
        for i, val in enumerate(v):
            nv = _rv(val, depth + 1)
            if nv is not val:
                if items is None:
                    items = list(v)
                items[i] = nv
        if items is None:
            return v
        return items if t is list else tuple(items)
    return _opaque(v, depth)


def _failure_event(event_dict: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {"event": REDACTION_FAILED, "redaction_error": True}
    with contextlib.suppress(Exception):
        lvl = event_dict.get("level")
        if isinstance(lvl, str) and lvl in _LEVEL_NAMES:
            out["level"] = lvl
        ts = event_dict.get("timestamp")
        if isinstance(ts, str):
            out["timestamp"] = ts
        lg = event_dict.get("logger")
        if isinstance(lg, str) and _PLAIN_LOGGER.match(lg):
            out["logger"] = lg
    return out


def redact_event_dict(
    logger: object, method_name: str, event_dict: Mapping[str, object]
) -> dict[str, object]:
    """structlog processor: return a redacted copy of ``event_dict``.

    MUST sit after every processor that turns objects into text
    (``format_exc_info``, ``StackInfoRenderer``) and immediately before the
    renderer.  Never raises.
    """
    try:
        ed = dict(event_dict)
        if "exc_info" in ed:
            try:
                from structlog.processors import format_exc_info

                ed = dict(format_exc_info(logger, method_name, ed))
            except Exception:
                _count_failure()
                ed.pop("exc_info", None)
                ed["exception"] = REDACTION_FAILED
        for k in list(ed):
            try:
                v = ed[k]
                if (
                    isinstance(k, str)
                    and v is not None
                    and type(v) is not bool
                    and (is_sensitive_key(k))
                ):
                    ed[k] = REDACTED
                    _count_hit("R-10")
                elif isinstance(v, str) and (
                    (k in PROCESSOR_KEYS and _SAFE_PROCESSOR_TEXT.match(v))
                    or (k == "event" and _PLAIN_EVENT.match(v))
                ):
                    # Processor-produced / plain identifier text: R-09 only.  A
                    # caller-supplied value under a processor key that does not look
                    # like processor output falls through to the full rules (CR-04).
                    r, n = _MATCHER.subn_all(v)
                    if n:
                        _count_hit("R-09", n)
                        ed[k] = r
                else:
                    nv = _rv(v, 0, head_tail=(k in HEAD_TAIL_KEYS))
                    if nv is not v:
                        ed[k] = nv
            except Exception:
                _count_failure()
                ed[k] = REDACTION_FAILED
        return ed
    except Exception:
        _count_failure()
        return _failure_event(event_dict)


# ---------------------------------------------------------------------------
# Renderer wrapper (final exact-match pass + fail-safe)
# ---------------------------------------------------------------------------


class RedactingRenderer:
    """Wrap the final renderer: fail-safe, then an exact-match pass over the line."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Callable[..., Any]) -> None:
        self._inner = inner

    @property
    def inner(self) -> Callable[..., Any]:
        return self._inner

    def __call__(
        self, logger: object, method_name: str, event_dict: Mapping[str, object]
    ) -> str | bytes | bytearray:
        try:
            out = self._inner(logger, method_name, event_dict)
        except Exception:
            _count_failure()
            return _FAILED_LINE
        try:
            if isinstance(out, str):
                return _MATCHER.sub_final(out)
            if isinstance(out, (bytes, bytearray)):
                # decode errors fall into the except below and fail closed
                return _MATCHER.sub_final(bytes(out).decode("utf-8")).encode("utf-8")
        except Exception:
            _count_failure()
        # a final renderer must produce text; anything else is a renderer fault
        return _FAILED_LINE
