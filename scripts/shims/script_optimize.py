"""Thin compatibility shim; implementation: ``scripts/entrypoints/optimize/script_optimize.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/optimize/script_optimize.py", globals())
