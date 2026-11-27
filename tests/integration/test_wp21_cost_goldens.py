"""
tests/integration/test_wp21_cost_goldens.py
--------------------------------------------
WP2.1a golden protocol (reports/vp2-wp2.1/synthesis-spec.md section 9.3).

    WP21-T-11  G-0 / G-1: scenarios A-D + SMK-T-09 under ``legacy_backtest_v0``
               reproduce ``LEGACY_DIGESTS`` byte for byte.  ``LEGACY_DIGESTS`` were
               recorded on PRISTINE HEAD 5b1151f with the pre-WP2.1 explicit kwargs
               ``maker_fee_bps=10, taker_fee_bps=15, slippage_bps=5`` and NO custom
               ``risk_params`` (CF-34: a custom ``RiskParameters`` hid the C25 rebuild
               in the old T-32 golden).  Scenario B is also the WP1.3b T-30 reference.
    WP21-T-12  G-1b: the T-32 one-cause re-pin (C25) is asserted in
               ``test_wp16_cb_backtest.py`` (explicit fill lists first, then the
               digest); here: it is deterministic and unaffected by the cost profile
               it pins (``legacy_backtest_v0``).
    WP21-T-13  G-2: per-profile goldens for ``coinbase_advanced_1`` and
               ``coinbase_advanced_d6`` with ANALYTIC per-fill checks, so a later
               switch of the default needs no re-pin.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pytest

from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.cost_scenarios import (
    SCENARIO_NAMES,
    digest,
    run_scenario,
    scenario_closes,
    scenario_initial_capital,
)
from tests.integration.test_wp13b_anchor_harness import GOLDEN_FIXED_DIGEST
from trading.costs import COST_PROFILES, CostProfile, get_cost_profile

#: Recorded on pristine HEAD 5b1151f (explicit 10/15/5 bps kwargs, default RiskParameters).
LEGACY_DIGESTS: dict[str, str] = {
    "A": "376f570753e5f9a46e12aed878a0f25ed22dbe34990019439f771a36206dcd2c",
    "B": "94015f75e15178a7d197488360f8cc72a0f6243cbf8dfde63c88fd46be3d3a5c",
    "C": "144c63a8ef6508a8d0bb3f4c2219d5c26e0deff5d3992788a54cda03fd5847ce",
    "D": "0e053bcaf2f836d938075bffcd3e2df8596ff37f4f7be119ac26c24ae8fe3ab6",
    "SMK": "7cf419f50c528e29cd5e9ae4de7530eeeed744523d05caf1282df0be1ebe24b8",
}

#: G-2 goldens, pinned PER PROFILE ID.  Scenario B under ``coinbase_advanced_1`` stops out on
#: the same bar as scenario A (X-15: the bracket basis is the all-in fill, so a 120 bps taker
#: pulls the 2% stop to ~0.7% below the market and it is hit at close 99 before the 4% TP leg),
#: which is why their digests are equal there -- they differ under every other profile.
PROFILE_DIGESTS: dict[str, dict[str, str]] = {
    "coinbase_advanced_1": {
        "A": "311ea61b5dcdd090b6eb59c62eb578c2708d450c05b1ad97a58375f0bb89f76f",
        "B": "311ea61b5dcdd090b6eb59c62eb578c2708d450c05b1ad97a58375f0bb89f76f",
        "C": "3f968ed58752ebf9a17d5ebc198c3fdc631a579877f1ad1dc517106727026aa2",
        "D": "907d897a7c109bcd93cb8b0de4a87db4bd302dd476e64cc21c3f941b98b08ad2",
        "SMK": "b21f47884be6d4f20a5316b4c54ffbdcb6e5c01d99e8b1036e4c67fede35e105",
    },
    "coinbase_advanced_d6": {
        "A": "edbf78b3778e930babf01d23fd98d6ddea0485fec4d8f2cfd9a60ff0c0d89127",
        "B": "62c4e7a8f4808e077b9e49dd3e45370b704a278a72b7cc8cb1c50b2ffe32be10",
        "C": "69c84aab3d40331b17442caf94934720c200168b721548f4995bfd4695a7b17f",
        "D": "fe19d51ea3ba63f006e552cc31632d22d20db1de512782a9c4b404076f794a95",
        "SMK": "b7cfa9a3b5ed5a4d28da26c91a624840176ad77e7d0c09a35a965035081c1fc5",
    },
}

_Q = Decimal("0.00000001")


# ---------------------------------------------------------------------------
# WP21-T-11 -- legacy byte-identity (G-0 / G-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIO_NAMES)
async def test_t11_legacy_backtest_profile_reproduces_the_head_digests(name: str) -> None:
    out = await run_scenario(name, cost_profile=get_cost_profile("legacy_backtest_v0"))
    assert digest(out) == LEGACY_DIGESTS[name]
    again = await run_scenario(name, cost_profile=get_cost_profile("legacy_backtest_v0"))
    assert digest(again) == LEGACY_DIGESTS[name]  # deterministic run to run


def test_t11_scenario_b_is_the_wp13b_t30_reference() -> None:
    assert LEGACY_DIGESTS["B"] == GOLDEN_FIXED_DIGEST


@pytest.mark.parametrize("name", ["A", "B", "C", "D", "SMK"])
async def test_t11_the_default_profile_is_not_the_legacy_golden(name: str) -> None:
    """The goldens are not vacuous: pricing under the default profile changes every digest."""
    out = await run_scenario(name)  # no profile -> DEFAULT
    assert digest(out) != LEGACY_DIGESTS[name]


# ---------------------------------------------------------------------------
# WP21-T-12 -- the T-32 re-pin is single-cause (see test_wp16_cb_backtest.py)
# ---------------------------------------------------------------------------


async def test_t12_t32_reference_is_pinned_to_the_legacy_backtest_profile() -> None:
    from tests.integration import test_wp16_cb_backtest as t32

    runner, _bars = t32._runner("reduce_daily", enable=False)
    assert runner.cost_profile.id == "legacy_backtest_v0"
    assert t32.GOLDEN_REFERENCE_DIGEST == (
        "5990ddc54bec8ff1bf9c3a46a52b66b3575cb4584b26d86650f7bca3bddae48d"
    )


# ---------------------------------------------------------------------------
# WP21-T-13 -- per-profile goldens with analytic per-fill checks (G-2)
# ---------------------------------------------------------------------------


def _expected_fill_price(expected: Decimal, side: str, profile: CostProfile) -> Decimal:
    factor = (
        Decimal(1) + profile.slippage_rate if side == "buy" else Decimal(1) - profile.slippage_rate
    )
    return (expected * factor).quantize(_Q, rounding=ROUND_HALF_UP)


@pytest.mark.parametrize("profile_id", sorted(PROFILE_DIGESTS))
@pytest.mark.parametrize("name", SCENARIO_NAMES)
async def test_t13_per_profile_golden_with_analytic_per_fill_checks(
    name: str, profile_id: str
) -> None:
    profile = COST_PROFILES[profile_id]
    out = await run_scenario(name, cost_profile=profile)
    closes = set(scenario_closes(name))

    equity = scenario_initial_capital(name)
    net_qty = Decimal(0)
    assert out["fills"], "the scenario must trade"
    for (side, qty_s, price_s), fee_s, expected_s in zip(
        out["fills"], out["fees"], out["expected"], strict=True
    ):
        qty, price, fee, expected = (
            Decimal(qty_s),
            Decimal(price_s),
            Decimal(fee_s),
            Decimal(expected_s),
        )
        assert expected in closes, "every fill is priced off a bar close"
        # price == quantize(close * (1 +/- slippage))
        assert price == _expected_fill_price(expected, side, profile)
        # fee == quantize(qty * price * taker, 8, HALF_UP)  (all orders are MARKET -> taker)
        assert fee == (qty * price * profile.fee_rate("taker")).quantize(_Q, rounding=ROUND_HALF_UP)
        if side == "buy":
            equity -= qty * price
            net_qty += qty
        else:
            equity += qty * price
            net_qty -= qty
        equity -= fee
    # final equity reconciles: cash flow + the open remainder marked at the last close
    equity += net_qty * scenario_closes(name)[-1]
    assert Decimal(out["final_equity"]) == equity

    assert digest(out) == PROFILE_DIGESTS[profile_id][name]


def test_t13_pins_exist_for_both_live_equivalent_profiles() -> None:
    live_equivalent = sorted(pid for pid, p in COST_PROFILES.items() if p.live_equivalent)
    assert sorted(PROFILE_DIGESTS) == live_equivalent
