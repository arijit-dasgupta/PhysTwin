"""Thin compatibility shim; implementation: ``scripts/entrypoints/eval/evaluate_chamfer.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/eval/evaluate_chamfer.py", globals())
