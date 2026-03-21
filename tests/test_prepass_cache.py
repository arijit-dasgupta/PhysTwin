"""prepass_cache roundtrip (no GPU)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rerun_viz.prepass_cache import (
    prepass_cache_file_path,
    save_prepass_cache,
    try_load_prepass_cache,
)
from rerun_viz.spring_mass_logging import GlobalColorRanges


def test_prepass_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    exp = tmp_path / "experiments" / "case_a" / "train"
    exp.mkdir(parents=True)
    ckpt = exp / "best_1.pth"
    ckpt.write_bytes(b"x")

    gr = GlobalColorRanges(
        stiffness=(1.0, 2.0),
        stretch=(0.9, 1.1),
        mass=(0.1, 0.2),
    )
    path = save_prepass_cache("case_a", 1, 10, str(ckpt), gr)
    assert path is not None
    assert path.is_file()

    loaded = try_load_prepass_cache("case_a", 1, 10, str(ckpt))
    assert loaded is not None
    assert loaded.stiffness == gr.stiffness
    assert loaded.stretch == gr.stretch
    assert loaded.mass == gr.mass


def test_prepass_cache_invalidates_on_checkpoint_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    exp = tmp_path / "experiments" / "case_b" / "train"
    exp.mkdir(parents=True)
    ckpt = exp / "best_1.pth"
    ckpt.write_bytes(b"v1")

    gr = GlobalColorRanges(stiffness=(1.0, 2.0), stretch=None, mass=None)
    save_prepass_cache("case_b", 1, 5, str(ckpt), gr)

    assert try_load_prepass_cache("case_b", 1, 5, str(ckpt)) is not None

    ckpt.write_bytes(b"v2-changed")
    # Ensure mtime differs from cached value (same-second writes can share mtime).
    st = os.stat(ckpt)
    os.utime(ckpt, (st.st_atime + 2.0, st.st_mtime + 2.0))
    assert try_load_prepass_cache("case_b", 1, 5, str(ckpt)) is None


def test_prepass_cache_file_path_naming() -> None:
    p = prepass_cache_file_path("my_case", 2, 99)
    assert p == Path("experiments/my_case/train/.replay_prepass_2_99.pkl")
