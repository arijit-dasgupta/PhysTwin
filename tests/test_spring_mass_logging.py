from __future__ import annotations

import pathlib
import sys

import numpy as np
import rerun as rr

# Ensure the project root is on sys.path so that `rerun_viz` can be imported
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rerun_viz.spring_mass_logging import (  # noqa: E402
    build_spring_strips,
    compute_spring_colors_from_stiffness,
    log_points_and_springs,
)


def test_build_spring_strips_basic():
    # Three object vertices and two controller vertices
    object_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    control_pos = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    # Springs connect: (obj0-obj1), (obj2-ctrl0), (ctrl0-ctrl1)
    springs = np.array(
        [
            [0, 1],
            [2, 3],
            [3, 4],
        ],
        dtype=np.int32,
    )

    strips = build_spring_strips(
        object_positions=object_pos,
        controller_positions=control_pos,
        springs=springs,
        num_object_points=object_pos.shape[0],
    )

    assert strips.shape == (3, 2, 3)
    # First strip endpoints should be object vertices 0 and 1
    np.testing.assert_allclose(strips[0, 0], object_pos[0])
    np.testing.assert_allclose(strips[0, 1], object_pos[1])
    # Second strip: object vertex 2 and control vertex 0
    np.testing.assert_allclose(strips[1, 0], object_pos[2])
    np.testing.assert_allclose(strips[1, 1], control_pos[0])
    # Third strip: control vertices 0 and 1
    np.testing.assert_allclose(strips[2, 0], control_pos[0])
    np.testing.assert_allclose(strips[2, 1], control_pos[1])


def test_build_spring_strips_no_controls():
    # Only object vertices, no controller vertices
    object_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    springs = np.array([[0, 1]], dtype=np.int32)

    strips = build_spring_strips(
        object_positions=object_pos,
        controller_positions=None,
        springs=springs,
        num_object_points=object_pos.shape[0],
    )

    assert strips.shape == (1, 2, 3)
    np.testing.assert_allclose(strips[0, 0], object_pos[0])
    np.testing.assert_allclose(strips[0, 1], object_pos[1])


def test_build_spring_strips_raises_on_out_of_bounds_index():
    object_pos = np.zeros((1, 3), dtype=np.float32)
    control_pos = np.zeros((1, 3), dtype=np.float32)
    # Index 5 is out of bounds for combined (2 vertices total)
    springs = np.array([[0, 5]], dtype=np.int32)

    try:
        build_spring_strips(
            object_positions=object_pos,
            controller_positions=control_pos,
            springs=springs,
            num_object_points=object_pos.shape[0],
        )
    except AssertionError:
        # Expected path
        return

    raise AssertionError("Expected build_spring_strips to raise AssertionError for bad index")


def test_compute_spring_colors_from_stiffness_range_and_shape():
    stiffness = np.array([0.1, 1.0, 10.0], dtype=np.float32)
    colors = compute_spring_colors_from_stiffness(stiffness)

    assert colors.shape == (3, 4)
    assert colors.dtype == np.uint8
    assert np.all(colors >= 0) and np.all(colors <= 255)


def test_compute_spring_colors_from_stiffness_empty():
    stiffness = np.array([], dtype=np.float32)
    colors = compute_spring_colors_from_stiffness(stiffness)
    assert colors.shape == (0, 4)


def test_compute_spring_colors_from_stiffness_monotonic_red_and_blue():
    # Increasing stiffness should generally increase red channel and decrease blue
    stiffness = np.array([0.1, 1.0, 10.0], dtype=np.float32)
    colors = compute_spring_colors_from_stiffness(stiffness)
    r = colors[:, 0].astype(int)
    b = colors[:, 2].astype(int)

    assert r[0] <= r[1] <= r[2]
    assert b[0] >= b[1] >= b[2]


def test_log_points_and_springs_runs_without_viewer():
    # This is a smoke test: ensure logging works when no viewer is connected.
    rr.init("test_spring_mass_logging", spawn=False, default_enabled=True)

    object_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    control_pos = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    springs = np.array([[0, 1]], dtype=np.int32)
    stiffness = np.array([1.0], dtype=np.float32)

    # Should not raise
    log_points_and_springs(
        object_positions=object_pos,
        controller_positions=control_pos,
        springs=springs,
        stiffness=stiffness,
        frame_idx=0,
        timeline="frame",
    )


def test_log_points_and_springs_handles_empty_springs_and_controls():
    rr.init("test_spring_mass_logging_empty", spawn=False, default_enabled=True)

    object_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    springs = np.zeros((0, 2), dtype=np.int32)
    stiffness = np.zeros((0,), dtype=np.float32)

    # Should not raise when there are no springs or controls
    log_points_and_springs(
        object_positions=object_pos,
        controller_positions=None,
        springs=springs,
        stiffness=stiffness,
        frame_idx=1,
        timeline="frame",
    )


