"""Helpers to pick a free TCP port for Rerun serve_grpc."""
from __future__ import annotations

import socket

# Rerun's serve_grpc binds like a typical server on all interfaces (IPv4).
# Checking only 127.0.0.1 can disagree with that and give false "busy" / "free"
# results on some systems (VM, SSH forwards, dual-stack).
_DEFAULT_CHECK_HOST = "0.0.0.0"


def port_free(port: int, *, host: str = _DEFAULT_CHECK_HOST) -> bool:
    """Return True if we can bind to ``host:port`` (same style as serve_grpc)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def pick_free_port(
    preferred: int,
    *,
    host: str = _DEFAULT_CHECK_HOST,
    max_scan: int = 50,
) -> tuple[int, bool]:
    """
    Return (port, used_fallback).

    If ``preferred`` is free, returns (preferred, False).
    Otherwise returns the first free port in [preferred+1, preferred+max_scan), with used_fallback True.
    """
    if port_free(preferred, host=host):
        return preferred, False
    for p in range(preferred + 1, preferred + max_scan):
        if port_free(p, host=host):
            return p, True
    raise RuntimeError(
        f"No free port found between {preferred} and {preferred + max_scan - 1} on {host}"
    )
