"""Thin compatibility shim; implementation: ``scripts/entrypoints/gaussian/gs_render.py``."""

from scripts._compat_shim import run_entrypoint

run_entrypoint(__name__, "entrypoints/gaussian/gs_render.py", globals())
