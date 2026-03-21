"""Tests for rerun_viz.port_util."""
from __future__ import annotations

import pathlib
import socket
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rerun_viz.port_util import pick_free_port, port_free


def test_pick_free_port_returns_preferred_when_free():
    # Use a high ephemeral port unlikely to be in use
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    preferred = s.getsockname()[1]
    s.close()

    p, fallback = pick_free_port(preferred)
    assert p == preferred
    assert fallback is False


def test_pick_free_port_skips_taken():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    taken = s.getsockname()[1]

    p, fallback = pick_free_port(taken)
    assert fallback is True
    assert p == taken + 1
    assert port_free(p)

    s.close()
