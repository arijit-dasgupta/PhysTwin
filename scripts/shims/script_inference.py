"""Thin compatibility shim; implementation: ``scripts/entrypoints/inference/script_inference.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/inference/script_inference.py", globals())
