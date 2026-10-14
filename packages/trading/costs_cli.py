"""
packages/trading/costs_cli.py
------------------------------
WP2.1: the ONE command-line surface for choosing a cost model in research
scripts (``--cost-profile`` plus the numeric ``--taker-bps`` / ``--maker-bps`` /
``--slippage-bps`` overrides).

Numeric overrides are LIBRARY / SCRIPT ONLY (SY-21-03): they produce an
``id="custom"`` / ``source="override"`` profile that is printed in the script
header but never persisted on a run row and never accepted by the API.

Kept out of ``trading.costs`` so that module stays a pure, environment-free,
argparse-free leaf.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation

from trading.costs import (
    COST_PROFILES,
    DEFAULT_COST_PROFILE_ID,
    CostProfile,
    default_cost_profile,
    get_cost_profile,
)

__all__ = ["add_cost_profile_args", "cost_header", "cost_profile_from_args"]


def _bps_arg(raw: str) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("must be a decimal number of basis points") from exc
    if not value.is_finite():
        raise argparse.ArgumentTypeError("must be finite")
    return value


def add_cost_profile_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--cost-profile`` and the three numeric override flags."""
    parser.add_argument(
        "--cost-profile",
        choices=sorted(COST_PROFILES),
        default=None,
        help=f"cost-profile registry id (default: {DEFAULT_COST_PROFILE_ID})",
    )
    parser.add_argument(
        "--taker-bps",
        type=_bps_arg,
        default=None,
        help="override the taker fee in bps (produces a 'custom' override profile)",
    )
    parser.add_argument(
        "--maker-bps",
        type=_bps_arg,
        default=None,
        help="override the maker fee in bps (produces a 'custom' override profile)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=_bps_arg,
        default=None,
        help="override the slippage in bps (produces a 'custom' override profile)",
    )


def cost_profile_from_args(args: argparse.Namespace) -> CostProfile:
    """Resolve the profile selected by the parsed ``args``.

    No flag -> the current default.  ``--cost-profile`` -> that registry id.
    Any numeric flag -> ``with_overrides`` of the base (``id="custom"``).
    Raises :class:`CostProfileError` on an invalid number.
    """
    base = (
        get_cost_profile(args.cost_profile)
        if args.cost_profile is not None
        else default_cost_profile()
    )
    taker = getattr(args, "taker_bps", None)
    maker = getattr(args, "maker_bps", None)
    slippage = getattr(args, "slippage_bps", None)
    if taker is None and maker is None and slippage is None:
        return base
    return base.with_overrides(taker_fee_bps=taker, maker_fee_bps=maker, slippage_bps=slippage)


def cost_header(profile: CostProfile) -> str:
    """One-line header printed by every script (id, fingerprint, source, rates)."""
    base = f" base={profile.base_id}" if profile.base_id else ""
    return (
        f"cost_profile id={profile.id} fingerprint={profile.fingerprint()} "
        f"source={profile.source}{base} maker={profile.maker_fee_bps}bps "
        f"taker={profile.taker_fee_bps}bps slippage={profile.slippage_bps}bps"
    )
