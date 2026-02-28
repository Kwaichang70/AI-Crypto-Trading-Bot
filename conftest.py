"""
conftest.py (project root)
--------------------------
Root-level pytest configuration.

Adds the workspace package directories to sys.path so that
``import common`` and ``import trading`` resolve correctly without
requiring an editable install.  This mirrors the uv workspace
package layout where each package lives under ``packages/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add workspace package roots to sys.path so that:
#   from common.types import OrderSide
#   from trading.models import Signal
# ... resolve without editable installs.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent
_PACKAGES_DIR = _REPO_ROOT / "packages"

# Insert package directories so all subpackages are discoverable
for _pkg_dir in [
    _PACKAGES_DIR / "common",
    _PACKAGES_DIR / "trading",
    _PACKAGES_DIR / "data",
]:
    _pkg_str = str(_pkg_dir.parent)  # add the 'packages/' directory itself
    if _pkg_str not in sys.path:
        sys.path.insert(0, _pkg_str)
