"""Pure helpers shared by replay and logging (no simulator)."""

from __future__ import annotations

import numpy as np

from rerun_viz.spring_mass_logging import (
    GlobalColorRanges,
    compute_global_color_ranges,
    object_object_spring_row_indices,
)


def test_object_object_spring_row_indices_basic():
    springs = np.array([[0, 1], [0, 2], [2, 3]], dtype=np.int32)  # 3 obj verts, last spring to ctrl
    idx = object_object_spring_row_indices(springs, num_object_vertices=3)
    np.testing.assert_array_equal(idx, np.array([0, 1], dtype=np.int64))


def test_object_object_spring_row_indices_empty():
    assert object_object_spring_row_indices(np.zeros((0, 2), dtype=np.int32), 5).size == 0


def test_compute_global_color_ranges_nonempty():
    stiffness = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    stretch = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    mass = np.array([0.1, 0.2], dtype=np.float32)
    gr = compute_global_color_ranges(stiffness, stretch, mass)
    assert isinstance(gr, GlobalColorRanges)
    assert gr.stiffness is not None and len(gr.stiffness) == 2
    assert gr.stretch is not None and len(gr.stretch) == 2
    assert gr.mass is not None and len(gr.mass) == 2


def test_compute_global_color_ranges_all_empty():
    gr = compute_global_color_ranges(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )
    assert gr.stiffness is None
    assert gr.stretch is None
    assert gr.mass is None


def test_compute_global_color_ranges_nonfinite_filtered():
    stiffness = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
    gr = compute_global_color_ranges(
        stiffness, np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    )
    assert gr.stiffness is not None
