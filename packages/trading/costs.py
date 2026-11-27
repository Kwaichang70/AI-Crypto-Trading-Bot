"""
packages/trading/costs.py
--------------------------
WP2.1 -- ONE cost model for every run mode.

``CostProfile`` is a frozen, Decimal-only description of the trading costs a
run is simulated (or, in live, *estimated*) under: maker / taker fee, adverse
slippage and an optional square-root impact term.  Every cost consumer
(``PaperExecutionEngine``, ``BacktestRunner``, ``ParameterOptimizer``,
``StrategyEngine``'s exit-cost floor, the live drift logs, the ML gate and the
research scripts) reads the SAME object, so backtest, paper and live use one
arithmetic.

Design rules (binding spec ``reports/vp2-wp2.1/synthesis-spec.md``)
------------------------------------------------------------------
* **Leaf module.**  Imports stdlib only (plus ``common.types`` if ever
  needed).  It reads **no environment**: the default profile is the code
  constant :data:`DEFAULT_COST_PROFILE_ID` (SY-21-02).  Switching it is a
  reviewed one-line change, never an env edit.
* **Registry is append-only.**  The numbers of an existing id are never
  changed -- add a new id instead.  Ids name the *tier*, never "default", so
  the default can move without renumbering.  Every fingerprint is pinned by a
  test.
* **Frozen per run.**  A run's profile is resolved once, persisted in
  ``RunORM.config["cost_profile"]`` and re-used on resume.  A run without the
  key is a *legacy* run and resolves to a legacy profile (never raises).
* **Live fees are the exchange's.**  In live the profile is only an estimate
  (exit floor, stop-cost log, drift warnings); the ledger uses real fills.
* ``maker_fee_bps`` is inert in every production path (all orders are MARKET).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from types import MappingProxyType
from typing import Final, Literal

__all__ = [
    "COST_PROFILES",
    "COST_PROFILE_SCHEMA",
    "DEFAULT_COST_PROFILE",
    "DEFAULT_COST_PROFILE_ID",
    "CostProfile",
    "CostProfileError",
    "CostProfileSource",
    "FeeSource",
    "Liquidity",
    "ResolvedCostProfile",
    "default_cost_profile",
    "get_cost_profile",
    "is_leaderboard_cost_eligible",
    "is_legacy_cost_config",
    "match_fee_tier",
    "new_run_cost_profile",
    "policy_side_cost_profile",
    "resolve_run_cost_profile",
]

_log = logging.getLogger(__name__)

COST_PROFILE_SCHEMA: Final = 1

Liquidity = Literal["maker", "taker"]
CostProfileSource = Literal["registry", "override", "legacy_inferred"]
FeeSource = Literal["model", "exchange_fills"]

_BPS: Final = Decimal(10_000)
_ID_RE: Final = re.compile(r"^[a-z0-9_]{3,48}$")
_PRICE_PRECISION: Final = Decimal("0.00000001")
_SOURCES: Final = frozenset({"registry", "override", "legacy_inferred"})
_FEE_SOURCES: Final = frozenset({"model", "exchange_fills"})
_CUSTOM_ID: Final = "custom"

#: Modes accepted by :func:`new_run_cost_profile` / :func:`resolve_run_cost_profile`.
_MODES: Final = frozenset({"backtest", "optimize", "paper", "live"})
#: The modes that run against the exchange (or its paper twin): a stored profile read
#: for one of these must be one a runtime writer can produce (WP2.1-H, CF-21-SEC-A).
_RUNTIME_MODES: Final = frozenset({"paper", "live"})


class CostProfileError(ValueError):
    """Validation, unknown id or strict-parse failure.

    ``code`` carries the stable API error code when there is one
    (``unknown_cost_profile``, ``cost_profile_not_live_equivalent``,
    ``cost_profile_not_allowed_for_live``).
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _check_bps(name: str, value: object, *, lo: Decimal, hi: Decimal) -> None:
    # ``bool`` / ``int`` / ``float`` / ``str`` are all rejected on purpose:
    # a float bps value would re-introduce binary rounding in the fee path.
    if not isinstance(value, Decimal):
        raise CostProfileError(f"{name} must be a Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise CostProfileError(f"{name} must be finite")
    if value < lo or value > hi:
        raise CostProfileError(f"{name} out of range [{lo}, {hi}]")
    if value != value.quantize(Decimal("0.01")):
        raise CostProfileError(f"{name} has more than 2 decimal places")


def _fmt(value: Decimal) -> str:
    """Canonical string of a Decimal (no exponent, no trailing zeros)."""
    return format(value.normalize(), "f")


@dataclass(frozen=True, slots=True)
class CostProfile:
    """Per-side trading costs, in basis points (Decimal only)."""

    id: str
    exchange: str
    tier: str
    maker_fee_bps: Decimal
    taker_fee_bps: Decimal
    slippage_bps: Decimal
    impact_coefficient_bps: Decimal = Decimal("0")
    fill_policy: Literal["market_taker"] = "market_taker"
    live_equivalent: bool = False
    diagnostic: bool = False
    legacy: bool = False
    stress: bool = False
    source: CostProfileSource = "registry"
    base_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_RE.fullmatch(self.id):
            raise CostProfileError("id must match ^[a-z0-9_]{3,48}$")
        if not isinstance(self.exchange, str) or not (0 < len(self.exchange) <= 48):
            raise CostProfileError("exchange must be a string of 1..48 characters")
        if not isinstance(self.tier, str) or not (0 < len(self.tier) <= 48):
            raise CostProfileError("tier must be a string of 1..48 characters")
        zero = Decimal(0)
        _check_bps("maker_fee_bps", self.maker_fee_bps, lo=zero, hi=Decimal(250))
        _check_bps("taker_fee_bps", self.taker_fee_bps, lo=zero, hi=Decimal(250))
        _check_bps("slippage_bps", self.slippage_bps, lo=zero, hi=Decimal(200))
        _check_bps("impact_coefficient_bps", self.impact_coefficient_bps, lo=zero, hi=Decimal(100))
        if self.maker_fee_bps > self.taker_fee_bps:
            raise CostProfileError("maker_fee_bps must be <= taker_fee_bps")
        if self.fill_policy != "market_taker":
            raise CostProfileError("fill_policy must be 'market_taker'")
        for flag in ("live_equivalent", "diagnostic", "legacy", "stress"):
            if not isinstance(getattr(self, flag), bool):
                raise CostProfileError(f"{flag} must be a bool")
        if sum((self.live_equivalent, self.diagnostic, self.legacy, self.stress)) > 1:
            raise CostProfileError(
                "at most one of live_equivalent/diagnostic/legacy/stress may be set"
            )
        if self.source not in _SOURCES:
            raise CostProfileError("source must be registry/override/legacy_inferred")
        if self.base_id is not None and (
            not isinstance(self.base_id, str) or not _ID_RE.fullmatch(self.base_id)
        ):
            raise CostProfileError("base_id must match ^[a-z0-9_]{3,48}$ or be None")

    # ------------------------------------------------------------------
    # Arithmetic (exact Decimal)
    # ------------------------------------------------------------------

    def fee_rate(self, liquidity: Liquidity) -> Decimal:
        """Fee as a fraction of fill notional (``bps / 10_000``, exact)."""
        if liquidity == "maker":
            return self.maker_fee_bps / _BPS
        if liquidity == "taker":
            return self.taker_fee_bps / _BPS
        raise CostProfileError(f"liquidity must be 'maker' or 'taker', got {liquidity!r}")

    @property
    def slippage_rate(self) -> Decimal:
        return self.slippage_bps / _BPS

    def side_cost_rate(self, liquidity: Liquidity = "taker") -> Decimal:
        """One side's cost: fee + slippage, as a fraction."""
        return self.fee_rate(liquidity) + self.slippage_rate

    def round_trip_cost_rate(self) -> Decimal:
        return 2 * self.side_cost_rate("taker")

    def estimate_fee(self, notional: Decimal, liquidity: Liquidity) -> Decimal:
        """ESTIMATE ONLY (never the ledger): ``notional * fee_rate``, 8 dp."""
        return (notional * self.fee_rate(liquidity)).quantize(
            _PRICE_PRECISION, rounding=ROUND_HALF_UP
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """16-hex digest of the economics (not of id / tier / flags)."""
        raw = "|".join(
            (
                "v1",
                _fmt(self.taker_fee_bps),
                _fmt(self.maker_fee_bps),
                _fmt(self.slippage_bps),
                _fmt(self.impact_coefficient_bps),
                self.fill_policy,
            )
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def with_overrides(
        self,
        *,
        taker_fee_bps: Decimal | None = None,
        maker_fee_bps: Decimal | None = None,
        slippage_bps: Decimal | None = None,
    ) -> CostProfile:
        """Library / script-only numeric override (never persisted on a run).

        Returns ``id="custom"``, ``source="override"``, ``base_id=self.id`` and
        ``live_equivalent=False``.
        """
        return replace(
            self,
            id=_CUSTOM_ID,
            source="override",
            base_id=self.id,
            live_equivalent=False,
            taker_fee_bps=self.taker_fee_bps if taker_fee_bps is None else taker_fee_bps,
            maker_fee_bps=self.maker_fee_bps if maker_fee_bps is None else maker_fee_bps,
            slippage_bps=self.slippage_bps if slippage_bps is None else slippage_bps,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_metadata(self, *, fee_source: FeeSource = "model") -> dict[str, object]:
        """JSON-safe dict stored in ``RunORM.config["cost_profile"]``."""
        if fee_source not in _FEE_SOURCES:
            raise CostProfileError("fee_source must be 'model' or 'exchange_fills'")
        return {
            "schema": COST_PROFILE_SCHEMA,
            "id": self.id,
            "exchange": self.exchange,
            "tier": self.tier,
            "maker_fee_bps": _fmt(self.maker_fee_bps),
            "taker_fee_bps": _fmt(self.taker_fee_bps),
            "slippage_bps": _fmt(self.slippage_bps),
            "impact_coefficient_bps": _fmt(self.impact_coefficient_bps),
            "fill_policy": self.fill_policy,
            "live_equivalent": self.live_equivalent,
            "diagnostic": self.diagnostic,
            "legacy": self.legacy,
            "stress": self.stress,
            "source": self.source,
            "base_id": self.base_id,
            "fingerprint": self.fingerprint(),
            "fee_source": fee_source,
        }

    @classmethod
    def from_metadata(cls, data: object) -> CostProfile:
        """Strict inverse of :meth:`to_metadata`; raises :class:`CostProfileError`.

        Schema-level only (an ``override`` profile round-trips).  Stored run rows
        are NOT read through here: they go through :func:`_parse_persisted`, which
        adds the source policy of WP2.1-H (CF-21-SEC-A).
        """
        return _parse_metadata(data)[0]


_META_KEYS: Final = frozenset(
    {
        "schema",
        "id",
        "exchange",
        "tier",
        "maker_fee_bps",
        "taker_fee_bps",
        "slippage_bps",
        "impact_coefficient_bps",
        "fill_policy",
        "live_equivalent",
        "diagnostic",
        "legacy",
        "stress",
        "source",
        "base_id",
        "fingerprint",
        "fee_source",
    }
)
_META_DECIMALS: Final = (
    "maker_fee_bps",
    "taker_fee_bps",
    "slippage_bps",
    "impact_coefficient_bps",
)
_META_BOOLS: Final = ("live_equivalent", "diagnostic", "legacy", "stress")


def _parse_metadata(data: object) -> tuple[CostProfile, FeeSource]:
    if not isinstance(data, Mapping):
        raise CostProfileError("cost_profile metadata must be an object")
    keys = set(data)
    if not all(isinstance(k, str) for k in keys):
        raise CostProfileError("cost_profile metadata has non-string keys")
    if keys != _META_KEYS:
        raise CostProfileError("cost_profile metadata keys do not match the schema")
    schema = data["schema"]
    if type(schema) is not int or schema != COST_PROFILE_SCHEMA:
        raise CostProfileError("unsupported cost_profile schema")
    decimals: dict[str, Decimal] = {}
    for key in _META_DECIMALS:
        raw = data[key]
        if not isinstance(raw, str):
            raise CostProfileError(f"{key} must be a decimal string")
        try:
            decimals[key] = Decimal(raw)
        except InvalidOperation as exc:
            raise CostProfileError(f"{key} is not a decimal string") from exc
    bools: dict[str, bool] = {}
    for key in _META_BOOLS:
        flag = data[key]
        if not isinstance(flag, bool):
            raise CostProfileError(f"{key} must be a bool")
        bools[key] = flag
    for key in ("id", "exchange", "tier", "fill_policy", "source", "fingerprint", "fee_source"):
        if not isinstance(data[key], str):
            raise CostProfileError(f"{key} must be a string")
    base_id = data["base_id"]
    if base_id is not None and not isinstance(base_id, str):
        raise CostProfileError("base_id must be a string or null")
    fee_source = data["fee_source"]
    if fee_source not in _FEE_SOURCES:
        raise CostProfileError("fee_source must be 'model' or 'exchange_fills'")
    if data["source"] not in _SOURCES:
        raise CostProfileError("source must be registry/override/legacy_inferred")
    if data["fill_policy"] != "market_taker":
        raise CostProfileError("fill_policy must be 'market_taker'")
    profile = CostProfile(
        id=str(data["id"]),
        exchange=str(data["exchange"]),
        tier=str(data["tier"]),
        maker_fee_bps=decimals["maker_fee_bps"],
        taker_fee_bps=decimals["taker_fee_bps"],
        slippage_bps=decimals["slippage_bps"],
        impact_coefficient_bps=decimals["impact_coefficient_bps"],
        fill_policy="market_taker",
        live_equivalent=bools["live_equivalent"],
        diagnostic=bools["diagnostic"],
        legacy=bools["legacy"],
        stress=bools["stress"],
        source=data["source"],
        base_id=base_id,
    )
    if profile.fingerprint() != data["fingerprint"]:
        raise CostProfileError("cost_profile fingerprint mismatch")
    if profile.source == "registry":
        registered = COST_PROFILES.get(profile.id)
        if registered is None:
            raise CostProfileError("registry cost_profile id is not in the registry")
        if registered != profile:
            raise CostProfileError("registry cost_profile differs from the registry entry")
    return profile, fee_source


def _parse_persisted(data: object, *, mode: str) -> tuple[CostProfile, FeeSource]:
    """Read a profile STORED on a run row (``RunORM.config`` / ``run_metadata``).

    WP2.1-H (CF-21-SEC-A, SEC-03, SEC-01).  On top of the schema checks of
    :func:`_parse_metadata`, a stored profile is accepted only if it is

    * ``source == "registry"`` (``_parse_metadata`` has already proven that the id
      is in the registry and that every field equals the registry entry) AND, for
      ``paper`` / ``live``, ``live_equivalent``.  Backtest / optimize accept any
      registry id (stress, diagnostic and legacy profiles are legitimate inputs
      there); paper and live accept none of those, because no writer stores one on
      a runtime run (``new_run_cost_profile``), or
    * ``source == "legacy_inferred"`` AND equal, field for field, to the legacy
      profile of the mode family (``legacy_runtime_v0`` for paper / live,
      ``legacy_backtest_v0`` for backtest / optimize) -- the one row the auto-retry
      writer persists.

    Anything else -- ``override``, another or hand-edited ``legacy_inferred`` row, a
    registry row that is not live-equivalent on a paper / live run -- is a tampered
    row.  A writer never produces one (``with_overrides`` is library / script only),
    so rejecting it cannot affect a legitimate run.  The accepted set for a mode is
    exactly the set its writers produce.  The error text carries no value (CF-36).
    """
    profile, fee_source = _parse_metadata(data)
    if profile.source == "registry":
        if mode in _RUNTIME_MODES and not profile.live_equivalent:
            raise CostProfileError("persisted cost_profile source is not accepted")
        return profile, fee_source
    if profile.source == "legacy_inferred" and profile == _legacy_for(mode):
        return profile, fee_source
    raise CostProfileError("persisted cost_profile source is not accepted")


# ---------------------------------------------------------------------------
# Registry (append-only; every fingerprint pinned by a test)
# ---------------------------------------------------------------------------


def _p(
    pid: str,
    exchange: str,
    tier: str,
    maker: int,
    taker: int,
    slip: int,
    *,
    live_equivalent: bool = False,
    diagnostic: bool = False,
    legacy: bool = False,
    stress: bool = False,
) -> CostProfile:
    return CostProfile(
        id=pid,
        exchange=exchange,
        tier=tier,
        maker_fee_bps=Decimal(maker),
        taker_fee_bps=Decimal(taker),
        slippage_bps=Decimal(slip),
        live_equivalent=live_equivalent,
        diagnostic=diagnostic,
        legacy=legacy,
        stress=stress,
    )


COST_PROFILES: Final[Mapping[str, CostProfile]] = MappingProxyType(
    {
        p.id: p
        for p in (
            # DEFAULT (U-1): Coinbase Advanced tier 1 (ccxt's own "Advanced 1").
            _p(
                "coinbase_advanced_1",
                "coinbase",
                "advanced_1_usd_0_1k_30d",
                60,
                120,
                10,
                live_equivalent=True,
            ),
            # D6: assumed 0-10k tier; WP2.8 base and grid point 60.
            _p(
                "coinbase_advanced_d6",
                "coinbase",
                "advanced_d6_assumed_0_10k",
                40,
                60,
                10,
                live_equivalent=True,
            ),
            # WP2.8 sensitivity grid: maker 40 / slip 10, only the taker varied.
            _p("coinbase_stress_t80", "coinbase", "stress_taker_80", 40, 80, 10, stress=True),
            _p("coinbase_stress_t100", "coinbase", "stress_taker_100", 40, 100, 10, stress=True),
            # Gross edge (backtest / optimize only).
            _p(
                "frictionless_diagnostic",
                "generic",
                "diagnostic_zero_cost",
                0,
                0,
                0,
                diagnostic=True,
            ),
            # Pre-WP2.1 behaviour, kept so old results stay reproducible.
            _p("legacy_backtest_v0", "generic", "pre_wp21_backtest", 10, 15, 5, legacy=True),
            _p("legacy_runtime_v0", "coinbase", "pre_wp21_runtime", 40, 60, 5, legacy=True),
        )
    }
)

#: THE one line changed after the operator's fee-tier verification (runbook L-21,
#: spec §8.4).  Never selected by environment (SY-21-02).
DEFAULT_COST_PROFILE_ID: Final[str] = "coinbase_advanced_1"
DEFAULT_COST_PROFILE: Final[CostProfile] = COST_PROFILES[DEFAULT_COST_PROFILE_ID]


def default_cost_profile() -> CostProfile:
    """The current default, read at call time (so tests may patch the module)."""
    return DEFAULT_COST_PROFILE


def get_cost_profile(profile_id: str) -> CostProfile:
    """Registry lookup; :class:`CostProfileError` (``unknown_cost_profile``) if absent."""
    profile = COST_PROFILES.get(profile_id) if isinstance(profile_id, str) else None
    if profile is None:
        raise CostProfileError("unknown cost profile id", code="unknown_cost_profile")
    return profile


def _check_mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}")
    return mode


def new_run_cost_profile(requested_id: str | None, *, mode: str) -> CostProfile:
    """Profile for a NEW run (U-2).

    ``None`` -> the default.  Backtest / optimize accept any registry id,
    paper only ``live_equivalent`` ids, live accepts no id at all.
    """
    _check_mode(mode)
    if mode == "live":
        if requested_id is not None:
            raise CostProfileError(
                "live runs do not accept a cost profile",
                code="cost_profile_not_allowed_for_live",
            )
        return default_cost_profile()
    if requested_id is None:
        return default_cost_profile()
    profile = get_cost_profile(requested_id)
    if mode == "paper" and not profile.live_equivalent:
        raise CostProfileError(
            "paper runs accept live-equivalent cost profiles only",
            code="cost_profile_not_live_equivalent",
        )
    return profile


@dataclass(frozen=True, slots=True)
class ResolvedCostProfile:
    profile: CostProfile
    persisted: bool  # False -> inferred (absent or invalid)
    invalid: bool  # a present but malformed value was ignored (one warning log)
    fee_source: FeeSource = "model"


def _legacy_for(mode: str) -> CostProfile:
    legacy_id = "legacy_backtest_v0" if mode in ("backtest", "optimize") else "legacy_runtime_v0"
    return replace(COST_PROFILES[legacy_id], source="legacy_inferred")


def resolve_run_cost_profile(
    config: Mapping[str, object] | None, *, mode: str, log_invalid: bool = True
) -> ResolvedCostProfile:
    """Profile of an EXISTING run, without writing anything.

    Never raises on data: a valid persisted profile -> ``persisted=True``;
    absent -> the legacy profile of the mode; malformed -> the legacy profile
    with ``invalid=True`` and ONE warning log (no value echoed -- CF-36;
    ``log_invalid=False`` for read-only response paths that resolve many rows).
    ``ValueError`` only for an invalid ``mode``.

    WP2.1-H (CF-21-SEC-A): a persisted profile is valid only if it is a registry
    profile or the mode family's own legacy profile (see :func:`_parse_persisted`);
    a tampered row (``override``, a foreign or edited ``legacy_inferred``, and for
    ``paper`` / ``live`` a registry row that is not ``live_equivalent``) reads as
    malformed.  So the paper / live resume basis is a live-equivalent registry
    profile or ``legacy_runtime_v0`` -- never below ``legacy_runtime_v0``.  It can
    still be ANY live-equivalent registry profile (an accepted residual: the
    stored id is not integrity-protected against a database writer).
    """
    _check_mode(mode)
    raw: object = None
    present = False
    if isinstance(config, Mapping) and "cost_profile" in config:
        present = True
        raw = config["cost_profile"]
    if not present or raw is None:
        return ResolvedCostProfile(_legacy_for(mode), persisted=False, invalid=False)
    try:
        profile, fee_source = _parse_persisted(raw, mode=mode)
    except CostProfileError:
        if log_invalid:
            _log.warning("cost_profile.invalid mode=%s action=fallback_to_legacy", mode)
        return ResolvedCostProfile(_legacy_for(mode), persisted=False, invalid=True)
    return ResolvedCostProfile(profile, persisted=True, invalid=False, fee_source=fee_source)


def is_legacy_cost_config(config: Mapping[str, object] | None, *, mode: str = "backtest") -> bool:
    """SY-21-10: key absent, or a ``legacy`` profile, or invalid.

    The answer is the ``legacy`` flag of the profile :func:`resolve_run_cost_profile`
    yields for ``mode``: a valid stored registry row gives its own flag, every other
    outcome (absent key, invalid row, the family's ``legacy_inferred`` row) is legacy.
    The default ``mode="backtest"`` accepts EVERY registry id (mode-agnostic callers);
    a caller that knows the run's mode passes it, so that a row the run's own reader
    rejects (WP2.1-H: a tampered ``override`` / foreign ``legacy_inferred`` row, or a
    paper / live registry row that is not live-equivalent) reads as legacy -- in step
    with ``cost_profile_invalid`` and the profile the response shows.
    """
    return resolve_run_cost_profile(config, mode=mode, log_invalid=False).profile.legacy


def is_leaderboard_cost_eligible(config: Mapping[str, object] | None) -> bool:
    """SY-21-10: the cost half of leaderboard eligibility -- a valid persisted
    ``live_equivalent`` REGISTRY profile that is not legacy.  Stress,
    diagnostic, override, legacy and invalid configs are never eligible.
    Never logs, never raises."""
    resolved = resolve_run_cost_profile(config, mode="backtest", log_invalid=False)
    profile = resolved.profile
    return (
        resolved.persisted
        and not resolved.invalid
        and profile.live_equivalent
        and profile.source == "registry"
        and not profile.legacy
    )


def policy_side_cost_profile(run_profile: CostProfile) -> CostProfile:
    """SY-21-06: of the default and the run's profile, the one with the larger
    taker side cost -- so an API pass implies an engine pass."""
    default = default_cost_profile()
    if run_profile.side_cost_rate("taker") > default.side_cost_rate("taker"):
        return run_profile
    return default


def match_fee_tier(taker_rate: object, maker_rate: object) -> str | None:
    """Registry id of the ``live_equivalent`` profile whose rates equal the
    exchange-reported ones (strings or Decimals, as fractions), else ``None``.

    Never raises: floats, ``None`` and garbage give ``None``.
    """

    def _bps(value: object) -> Decimal | None:
        if isinstance(value, bool) or not isinstance(value, str | Decimal):
            return None
        try:
            rate = Decimal(value.strip() if isinstance(value, str) else value)
            if not rate.is_finite():
                return None
            return rate * _BPS
        except ArithmeticError:  # InvalidOperation, Overflow (WP21A-CR-03/SEC-03): never raises
            return None

    taker_bps = _bps(taker_rate)
    maker_bps = _bps(maker_rate)
    if taker_bps is None or maker_bps is None:
        return None
    for profile in COST_PROFILES.values():
        if (
            profile.live_equivalent
            and profile.taker_fee_bps == taker_bps
            and profile.maker_fee_bps == maker_bps
        ):
            return profile.id
    return None
