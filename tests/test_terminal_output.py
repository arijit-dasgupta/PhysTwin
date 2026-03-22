"""Tests for rerun_viz.terminal_output (TTY vs non-TTY)."""

from __future__ import annotations

import io
import re

from rerun_viz.terminal_output import print_dim_status, print_file_recording, print_replay_banner


def test_print_replay_banner_no_ansi_when_not_tty():
    buf = io.StringIO()
    print_replay_banner("c1", "file", stream=buf)
    out = buf.getvalue()
    assert "PhysTwin" in out
    assert "\033[" not in out


def test_print_file_recording_no_ansi_when_not_tty():
    buf = io.StringIO()
    print_file_recording("/tmp/x.rrd", stream=buf)
    out = buf.getvalue()
    assert "/tmp/x.rrd" in out
    assert "\033[" not in out


def test_print_dim_status_no_ansi_when_not_tty():
    buf = io.StringIO()
    print_dim_status("hello", stream=buf)
    assert "hello" in buf.getvalue()
    assert "\033[" not in buf.getvalue()


def test_print_replay_banner_includes_ansi_when_tty(monkeypatch):
    buf = io.StringIO()

    def fake_isatty() -> bool:
        return True

    monkeypatch.setattr(buf, "isatty", fake_isatty)
    print_replay_banner("case", "serve", stream=buf)
    out = buf.getvalue()
    assert re.search(r"\033\[[0-9;]*m", out)
