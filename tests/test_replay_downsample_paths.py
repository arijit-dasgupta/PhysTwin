"""Downsampled replay path resolution (mocks heavy trainer GPU init)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

pytest.importorskip("rerun_viz.replay_core")
pytest.importorskip("warp")

from downsampling.validate import downsampled_bundle_dir


def _write_minimal_downsample_bundle(base: Path, case: str, tag: str) -> None:
    root = Path(downsampled_bundle_dir(str(base), case, tag))
    root.mkdir(parents=True)
    K, C, N_fine, T = 3, 2, 5, 2
    obj = np.random.randn(T, K, 3).astype(np.float32)
    ctrl = np.random.randn(T, C, 3).astype(np.float32)
    init_vertices = np.vstack([obj[0].astype(np.float64), ctrl[0].astype(np.float64)])
    springs = np.array([[0, 1], [1, 2], [K + 0, 0], [K + 1, 1]], dtype=np.int32)
    E = springs.shape[0]
    np.savez(
        root / "coarse_model.npz",
        init_vertices=init_vertices,
        init_springs=springs,
        init_rest_lengths=np.full(E, 0.05, dtype=np.float64),
        init_masses=np.ones(K + C, dtype=np.float64),
        num_object_springs=np.int32(2),
        num_all_points=np.int32(K),
    )
    np.save(root / "partition_map.npy", np.arange(N_fine, dtype=np.int32) % K)
    (root / "downsample_meta.json").write_text(
        json.dumps({"version": 1, "K": K, "method": "kmeans", "r": 2.0, "N_fine": N_fine})
    )
    with open(root / "final_data_downsampled.pkl", "wb") as f:
        pickle.dump(
            {
                "object_points": obj,
                "object_colors": np.ones((T, K, 3), dtype=np.float32),
                "object_visibilities": np.ones((T, K), dtype=bool),
                "object_motions_valid": np.ones((T, K), dtype=bool),
                "controller_points": ctrl,
                "surface_points": np.zeros((0, 3), dtype=np.float32),
                "interior_points": np.zeros((0, 3), dtype=np.float32),
            },
            f,
        )
    torch.save(
        {
            "epoch": 0,
            "num_object_springs": 2,
            "spring_Y": torch.ones(E, dtype=torch.float32) * 1000.0,
            "collide_elas": torch.tensor(0.1),
            "collide_fric": torch.tensor(0.2),
            "collide_object_elas": torch.tensor(0.3),
            "collide_object_fric": torch.tensor(0.4),
        },
        root / "best_downsampled.pth",
    )


def test_load_trainer_downsample_passes_precomputed_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import rerun_viz.replay_core as rc

    case, tag = "c1", "t1"
    _write_minimal_downsample_bundle(tmp_path, case, tag)

    captured: dict = {}

    def fake_trainer_ctor(*args, **kwargs):
        captured.update(kwargs)
        m = MagicMock()
        m.simulator.n_springs = 4
        return m

    monkeypatch.setattr(rc, "InvPhyTrainerWarp", fake_trainer_ctor)
    monkeypatch.setattr(rc, "_apply_spring_mass_checkpoint", lambda _t, _p: None)

    rc.load_trainer_and_model(
        str(tmp_path),
        case,
        downsample_version=tag,
        emit_downsample_tty_summary=False,
    )

    dp = captured.get("data_path", "")
    assert str(dp).endswith("final_data_downsampled.pkl"), dp
    pg = captured.get("precomputed_graph")
    assert pg is not None
    assert "init_vertices" in pg and "init_springs" in pg
    assert captured.get("pure_inference_mode") is True
