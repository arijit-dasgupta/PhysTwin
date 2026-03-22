"""Thin compatibility shim; implementation: ``scripts/entrypoints/playground/run_playground_gradio.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/playground/run_playground_gradio.py", globals())
