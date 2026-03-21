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
    compute_spring_colors_from_stretch,
    log_forces,
    log_points_and_springs,
    log_springs_by_stretch,
    log_velocities,
    object_object_spring_row_indices,
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


def test_object_object_spring_row_indices_matches_stretch_filter():
    """Same rows as log_springs_by_stretch uses for object–object springs."""
    springs = np.array([[0, 1], [1, 4], [2, 3]], dtype=np.int32)
    idx = object_object_spring_row_indices(springs, num_object_vertices=3)
    np.testing.assert_array_equal(idx, np.array([0], dtype=np.int64))


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


def test_compute_spring_colors_from_stretch_comparative():
    """With percentile norm: smallest ratio -> dark red, largest -> bright orange."""
    ratios = np.array([0.5, 0.8, 1.2], dtype=np.float32)
    colors = compute_spring_colors_from_stretch(ratios)
    assert colors.shape == (3, 4)
    assert colors.dtype == np.uint8
    # Highest stretch ratio should be brighter / more orange (higher R than lowest)
    assert colors[2, 0] > colors[0, 0]
    assert colors[2, 1] > colors[0, 1]


def test_compute_spring_colors_from_stretch_uniform():
    """When all ratios equal, get mid color."""
    ratios = np.array([1.0, 1.0], dtype=np.float32)
    colors = compute_spring_colors_from_stretch(ratios)
    assert colors.shape == (2, 4)
    assert np.allclose(colors[0], colors[1])


def test_compute_spring_colors_from_stretch_empty():
    ratios = np.array([], dtype=np.float32)
    colors = compute_spring_colors_from_stretch(ratios)
    assert colors.shape == (0, 4)


def test_collision_pair_extraction():
    """Extract (i,j) pairs from collision_indices/collision_number, avoiding duplicates."""
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        dtype=np.float32,
    )
    # Simulate: point 0 has neighbors 1,2; point 1 has neighbor 0; point 2 has neighbor 0
    # collision_indices[i, k] = neighbor index. We only add when i < j to avoid (i,j) and (j,i).
    collision_indices = np.zeros((3, 500), dtype=np.int32)
    collision_number = np.array([2, 1, 1], dtype=np.int32)
    collision_indices[0, 0] = 1
    collision_indices[0, 1] = 2
    collision_indices[1, 0] = 0
    collision_indices[2, 0] = 0

    from rerun_viz.spring_mass_logging import log_collisions

    rr.init("test_collisions", spawn=False, default_enabled=True)
    rr.set_time("frame", sequence=0)
    log_collisions(positions, collision_indices, collision_number)
    # Should not raise; we're testing the logic runs. Exact strip count: (0,1), (0,2) since we add when i<j.


def test_log_velocities_arrows():
    """log_velocities produces Arrows3D from positions and scaled vectors."""
    rr.init("test_velocities", spawn=False, default_enabled=True)
    rr.set_time("frame", sequence=0)
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    velocities = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    log_velocities(positions, velocities, scale=0.1)
    # Should not raise


def test_log_forces_arrows():
    """log_forces produces Arrows3D from positions and scaled vectors."""
    rr.init("test_forces", spawn=False, default_enabled=True)
    rr.set_time("frame", sequence=0)
    positions = np.array([[0, 0, 0]], dtype=np.float32)
    forces = np.array([[100, 0, 0]], dtype=np.float32)
    log_forces(positions, forces, scale=1e-4)
    # Should not raise


def test_log_springs_by_stretch_smoke():
    """log_springs_by_stretch runs without error for simple inputs."""
    rr.init("test_stretch", spawn=False, default_enabled=True)
    rr.set_time("frame", sequence=0)
    object_pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    springs = np.array([[0, 1]], dtype=np.int32)
    rest_lengths = np.array([1.0], dtype=np.float32)
    log_springs_by_stretch(
        object_positions=object_pos,
        controller_positions=None,
        springs=springs,
        rest_lengths=rest_lengths,
    )
