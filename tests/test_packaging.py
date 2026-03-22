"""Packaging / import layout (editable install)."""

from __future__ import annotations

import importlib
import importlib.util


def test_top_level_package_specs():
    """Do not ``import qqtt`` here: it pulls gaussian_splatting (needs built simple_knn)."""
    for name in ("qqtt", "rerun_viz", "benchmarks"):
        spec = importlib.util.find_spec(name)
        assert spec is not None, name


def test_package_locations_under_src_or_repo():
    qqtt_spec = importlib.util.find_spec("qqtt")
    rerun_spec = importlib.util.find_spec("rerun_viz")
    benchmarks = importlib.import_module("benchmarks")

    assert qqtt_spec.origin and "src" in qqtt_spec.origin.replace("\\", "/")
    assert rerun_spec.origin and "src" in rerun_spec.origin.replace("\\", "/")
    assert "benchmarks" in benchmarks.__file__.replace("\\", "/")


def test_rerun_viz_replay_recorded_module_exists():
    """Avoid importing replay (pulls trainer + gaussian) in CPU-only CI."""
    spec = importlib.util.find_spec("rerun_viz.replay_recorded")
    assert spec is not None


def test_import_qqtt_when_simple_knn_available():
    """Full import parity check when vendored gaussian extensions are installed."""
    try:
        import simple_knn  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("simple_knn extension not built (see gaussian_splatting/submodules/simple-knn)")

    import qqtt  # noqa: F401

    assert hasattr(qqtt, "InvPhyTrainerWarp")
