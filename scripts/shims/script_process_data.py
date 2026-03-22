"""Thin compatibility shim; implementation: ``scripts/entrypoints/data/script_process_data.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/data/script_process_data.py", globals())
