"""
packages/trading/_schema_utils.py
----------------------------------
Internal JSON-Schema post-processing helpers.

Used by strategy classes to normalise Pydantic v2's default schema output
for the optimizer UI form-builder (which expects a single scalar ``type``
field, not the ``anyOf`` union or list-form ``type`` that Pydantic emits
for nullable fields like ``trailing_stop_pct: float | None``).

Motivation (INF-5)
------------------
Pydantic v2's ``model_json_schema()`` emits one of two shapes for
``float | None = Field(ge=x, le=y)``:

    anyOf-form (observed in Pydantic 2.x):
        {"anyOf": [{"type": "number", "minimum": x, "maximum": y}, {"type": "null"}],
         "default": null, ...}

    list-form (JSON Schema draft 2020-12 alternative, not observed but handled
    defensively):
        {"type": ["number", "null"], "minimum": x, "maximum": y}

Both forms lack a single top-level ``"type"`` string, which breaks the
frontend form-builder's ``schema.type`` access.

This helper collapses both forms into::

    {"type": "number", "nullable": true, "minimum": x, "maximum": y, ...rest}

The ``"nullable": true`` flag is a non-standard extension understood by the
optimizer UI; it is not part of JSON Schema proper but is widely used by
OpenAPI 3.0 / JSON Schema tooling.

The underscore prefix signals these are package-internal helpers; external
consumers should use the public ``parameter_schema()`` classmethods on each
strategy.

Currently exports
-----------------
- ``normalise_nullable_json_schema``: recursive schema post-processor.
"""

from __future__ import annotations

import copy
from typing import Any


def normalise_nullable_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively collapse nullable unions into ``{type, nullable: true}``.

    Handles the ``properties`` sub-dict of a top-level JSON Schema object,
    transforming each property value that matches one of the nullable patterns.

    Parameters
    ----------
    schema:
        A JSON Schema dict as returned by Pydantic v2's
        ``BaseModel.model_json_schema()``.  Must be a ``dict``; non-dict
        values pass through unchanged.

    Returns
    -------
    dict[str, Any]
        A deep-copied, transformed schema.  The input is never mutated.

    Notes
    -----
    Idempotent — calling this function twice on the same input yields
    identical output to calling it once.

    Patterns collapsed
    ------------------
    1. ``anyOf`` form (Pydantic v2 default for ``T | None``)::

           {"anyOf": [<T-schema>, {"type": "null"}], ...rest}
           → {**<T-schema>, "nullable": True, ...rest}

       where ``<T-schema>`` is any schema object whose ``type`` is not
       ``"null"``.  ``"anyOf"`` and ``"type"`` (if absent in ``rest``) are
       not carried forward.

    2. List-form ``type`` (JSON Schema draft 2020-12)::

           {"type": ["T", "null"], ...rest}
           → {"type": "T", "nullable": True, ...rest}

    Non-nullable properties and other schema shapes pass through unchanged.
    """
    schema = copy.deepcopy(schema)
    transformed: dict[str, Any] = _transform(schema)
    return transformed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_null_schema(s: Any) -> bool:
    """Return True iff *s* is exactly ``{"type": "null"}``."""
    return isinstance(s, dict) and s.get("type") == "null" and len(s) == 1


def _collapse_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Collapse a single property schema if it matches a nullable pattern."""
    # Pattern 1: anyOf [T, null]
    any_of = prop.get("anyOf")
    if isinstance(any_of, list) and len(any_of) == 2:
        non_null = [s for s in any_of if not _is_null_schema(s)]
        null_entries = [s for s in any_of if _is_null_schema(s)]
        if len(non_null) == 1 and len(null_entries) == 1:
            t_schema = non_null[0]
            if isinstance(t_schema, dict) and "type" in t_schema:
                # Build merged dict: start with non-anyOf top-level keys,
                # overlay T-schema keys, add nullable flag.
                merged: dict[str, Any] = {
                    k: v for k, v in prop.items() if k != "anyOf"
                }
                merged.update(t_schema)
                merged["nullable"] = True
                return merged

    # Pattern 2: type: ["T", "null"] (list-form)
    type_val = prop.get("type")
    if (
        isinstance(type_val, list)
        and len(type_val) == 2
        and "null" in type_val
    ):
        scalar = next((t for t in type_val if t != "null"), None)
        if scalar is not None:
            result = dict(prop)
            result["type"] = scalar
            result["nullable"] = True
            return result

    return prop


def _transform(node: Any) -> Any:
    """Recursively transform a schema node."""
    if not isinstance(node, dict):
        return node

    # If this node has a ``properties`` sub-dict, transform each property.
    if "properties" in node and isinstance(node["properties"], dict):
        node["properties"] = {
            k: _collapse_property(_transform(v))
            for k, v in node["properties"].items()
        }

    # Recurse into other dict values that may be nested schemas.
    # (covers $defs, allOf items, etc. — not needed for current strategies
    #  but makes the helper forward-compatible.)
    for key, value in list(node.items()):
        if key == "properties":
            continue  # already handled above
        if isinstance(value, dict):
            node[key] = _transform(value)
        elif isinstance(value, list):
            node[key] = [_transform(item) for item in value]

    return node
