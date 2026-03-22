"""Thin compatibility shim; implementation: ``scripts/entrypoints/train/script_train.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/train/script_train.py", globals())
