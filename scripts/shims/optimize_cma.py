"""Thin compatibility shim; implementation: ``scripts/entrypoints/optimize/optimize_cma.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/optimize/optimize_cma.py", globals())
