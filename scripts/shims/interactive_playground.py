"""Thin compatibility shim; implementation: ``scripts/entrypoints/playground/interactive_playground.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/playground/interactive_playground.py", globals())
