"""Shared pytest configuration.

Makes the project root importable so `src` and `experiments` resolve during
tests. `pyproject.toml` also sets `pythonpath = ["."]` for pytest 7+, so this
is belt-and-braces for older pytest versions and for direct `python tests/...`
invocations.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
