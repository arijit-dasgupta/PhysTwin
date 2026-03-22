"""Thin compatibility shim; implementation: ``scripts/entrypoints/inference/inference_warp.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/inference/inference_warp.py", globals())
