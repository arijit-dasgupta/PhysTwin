"""Helpers for thin root shims that delegate to ``scripts/entrypoints/`` (internal)."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def run_entrypoint(caller_name: str, rel: str, globs: dict[str, object]) -> None:
    """If executed as ``__main__``, run the implementation with ``runpy``; else load and re-export."""
    impl = _REPO_ROOT / "scripts" / rel
    if caller_name == "__main__":
        runpy.run_path(str(impl), run_name="__main__")
        return
    spec = importlib.util.spec_from_file_location("_phystwin_entry_impl", impl)
    if spec is None or spec.loader is None:
        msg = f"Cannot load entrypoint {impl}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if not name.startswith("_"):
            globs[name] = getattr(mod, name)
