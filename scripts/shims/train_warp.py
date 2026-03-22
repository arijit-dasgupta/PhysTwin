"""Thin compatibility shim; implementation: ``scripts/entrypoints/train/train_warp.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/train/train_warp.py", globals())
