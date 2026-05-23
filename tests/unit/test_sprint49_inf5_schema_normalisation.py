"""
Sprint 49 / INF-5 — unit tests for _schema_utils.normalise_nullable_json_schema.

Test surface
------------
1.  anyOf [number, null] → {type: number, nullable: True, constraints preserved}
2.  anyOf [null, number] (reversed order) — same result
3.  list-form type: ["integer", "null"] → {type: integer, nullable: True}
4.  Non-nullable field passes through unchanged
5.  Multiple properties — only nullable ones transformed
6.  Idempotent: running twice = same output
7.  Deep-copy: input not mutated
8.  Constraints (minimum/maximum/exclusiveMinimum) survive collapse
9.  Non-anyOf properties (description, default, title) survive collapse
10. Integration: _BreakoutParams.model_json_schema() after normalisation has
    trailing_stop_pct with type="number" and nullable=True
"""

from __future__ import annotations

import copy
import sys
import os

import pytest

# ---------------------------------------------------------------------------
# Path setup — mirrors other Sprint 49 test files
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps"))

from trading._schema_utils import normalise_nullable_json_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema(prop: dict) -> dict:
    """Wrap a property dict into a minimal JSON Schema object."""
    return {"type": "object", "properties": {"field": prop}}


def _extract(normalised: dict) -> dict:
    return normalised["properties"]["field"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnyOfCollapse:
    """anyOf [T, null] → {type, nullable: true} with constraints."""

    def test_number_null_anyof(self):
        raw = _make_schema({
            "anyOf": [
                {"type": "number", "minimum": 0.005, "maximum": 0.5},
                {"type": "null"},
            ],
            "default": None,
            "title": "Trailing Stop Pct",
            "description": "Trailing stop percentage.",
        })
        result = _extract(normalise_nullable_json_schema(raw))
        assert result["type"] == "number"
        assert result["nullable"] is True
        assert result["minimum"] == 0.005
        assert result["maximum"] == 0.5
        assert result["default"] is None
        assert result["title"] == "Trailing Stop Pct"
        assert "anyOf" not in result

    def test_reversed_anyof_order(self):
        """null entry first — same result."""
        raw = _make_schema({
            "anyOf": [
                {"type": "null"},
                {"type": "integer", "minimum": 1, "maximum": 100},
            ],
            "default": None,
        })
        result = _extract(normalise_nullable_json_schema(raw))
        assert result["type"] == "integer"
        assert result["nullable"] is True
        assert result["minimum"] == 1
        assert "anyOf" not in result

    def test_string_nullable(self):
        raw = _make_schema({
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
        })
        result = _extract(normalise_nullable_json_schema(raw))
        assert result["type"] == "string"
        assert result["nullable"] is True

    def test_constraints_preserved(self):
        """exclusiveMinimum and other constraints must survive."""
        raw = _make_schema({
            "anyOf": [
                {"type": "number", "exclusiveMinimum": 0.0, "maximum": 10.0},
                {"type": "null"},
            ],
        })
        result = _extract(normalise_nullable_json_schema(raw))
        assert result["exclusiveMinimum"] == 0.0
        assert result["maximum"] == 10.0


class TestListFormCollapse:
    """type: [T, "null"] list-form → {type: T, nullable: true}."""

    def test_integer_null_list(self):
        raw = _make_schema({
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 500,
        })
        result = _extract(normalise_nullable_json_schema(raw))
        assert result["type"] == "integer"
        assert result["nullable"] is True
        assert result["minimum"] == 1

    def test_number_null_list(self):
        raw = _make_schema({"type": ["number", "null"]})
        result = _extract(normalise_nullable_json_schema(raw))
        assert result["type"] == "number"
        assert result["nullable"] is True


class TestPassThrough:
    """Non-nullable and non-matching schemas pass through unchanged."""

    def test_plain_integer(self):
        prop = {"type": "integer", "minimum": 2, "maximum": 500, "default": 20}
        raw = _make_schema(prop)
        result = _extract(normalise_nullable_json_schema(raw))
        assert result == prop

    def test_plain_boolean(self):
        prop = {"type": "boolean", "default": False}
        raw = _make_schema(prop)
        result = _extract(normalise_nullable_json_schema(raw))
        assert result == prop

    def test_anyof_non_null_not_collapsed(self):
        """anyOf with two non-null types must NOT be collapsed."""
        prop = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
        raw = _make_schema(prop)
        result = _extract(normalise_nullable_json_schema(raw))
        assert "anyOf" in result
        assert "nullable" not in result


class TestIdempotentAndImmutable:
    """Running twice = same output; input not mutated."""

    def test_idempotent(self):
        raw = _make_schema({
            "anyOf": [
                {"type": "number", "minimum": 0.005, "maximum": 0.5},
                {"type": "null"},
            ],
            "default": None,
        })
        once = normalise_nullable_json_schema(raw)
        twice = normalise_nullable_json_schema(once)
        assert once == twice

    def test_does_not_mutate_input(self):
        raw = _make_schema({
            "anyOf": [
                {"type": "number", "minimum": 0.005},
                {"type": "null"},
            ],
        })
        original = copy.deepcopy(raw)
        normalise_nullable_json_schema(raw)
        assert raw == original


class TestMultipleProperties:
    """Only nullable properties are transformed; others pass through."""

    def test_mixed_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "lookback_period": {"type": "integer", "minimum": 2, "default": 20},
                "position_size": {"type": "number", "exclusiveMinimum": 0.0},
                "trailing_stop_pct": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.005, "maximum": 0.5},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
        }
        result = normalise_nullable_json_schema(schema)
        props = result["properties"]

        # Non-nullable fields unchanged
        assert props["lookback_period"] == {"type": "integer", "minimum": 2, "default": 20}
        assert props["position_size"] == {"type": "number", "exclusiveMinimum": 0.0}

        # Nullable field collapsed
        tsp = props["trailing_stop_pct"]
        assert tsp["type"] == "number"
        assert tsp["nullable"] is True
        assert tsp["minimum"] == 0.005
        assert tsp["maximum"] == 0.5
        assert tsp["default"] is None
        assert "anyOf" not in tsp


class TestIntegration:
    """_BreakoutParams.model_json_schema() → normalised trailing_stop_pct."""

    def test_breakout_schema_trailing_stop(self):
        from trading.strategies.breakout import _BreakoutParams
        raw = _BreakoutParams.model_json_schema()
        result = normalise_nullable_json_schema(raw)
        tsp = result["properties"]["trailing_stop_pct"]
        assert tsp["type"] == "number", (
            f"Expected type='number', got {tsp.get('type')!r} — "
            "Pydantic v2 output shape may have changed"
        )
        assert tsp["nullable"] is True
        assert "anyOf" not in tsp
        assert tsp["minimum"] == 0.005
        assert tsp["maximum"] == 0.5
