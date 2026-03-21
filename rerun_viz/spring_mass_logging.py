from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_SPRING_LEGEND_LOGGED = False
import rerun as rr
import torch
import warp as wp


def _to_numpy_vec3(wp_array: wp.array, requires_grad: bool = False) -> np.ndarray:
    """Convert a Warp vec3 array to a contiguous (N, 3) numpy array."""
    t = wp.to_torch(wp_array, requires_grad=requires_grad)
    # Always detach & move to CPU for logging
    return t.detach().cpu().numpy()


def build_spring_strips(
    object_positions: np.ndarray,
    controller_positions: Optional[np.ndarray],
    springs: np.ndarray,
    num_object_points: int,
) -> np.ndarray:
    """Build (n_springs, 2, 3) strips in world-space from topology and positions.

    This is a pure-numpy helper that does not depend on Warp and is unit-testable.
    """
    assert object_positions.ndim == 2 and object_positions.shape[1] == 3
    assert springs.ndim == 2 and springs.shape[1] == 2

    if controller_positions is not None:
        assert controller_positions.ndim == 2 and controller_positions.shape[1] == 3
        combined = np.concatenate([object_positions, controller_positions], axis=0)
    else:
        combined = object_positions

    # Ensure indices are in range
    max_index = springs.max(initial=-1)
    assert max_index < combined.shape[0], (
        f"Spring endpoint index {max_index} out of bounds for "
        f"{combined.shape[0]} total vertices (object + control)."
    )

    # Each spring becomes a 2-point line strip
    strips = combined[springs.astype(np.int64)]  # (n_springs, 2, 3)
    return strips


def compute_spring_colors_from_stiffness(stiffness: np.ndarray) -> np.ndarray:
    """Map per-spring stiffness values to RGBA colors.

    Blue (soft) → purple (mid) → red (stiff)
    """
    if stiffness.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)

    vals = np.asarray(stiffness, dtype=np.float32)
    v_min = float(vals.min())
    v_max = float(vals.max())
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        vals = np.zeros_like(vals)
        v_min, v_max = 0.0, 1.0

    denom = max(v_max - v_min, 1e-8)
    x = (vals - v_min) / denom  # in [0, 1]

    # Blue (soft) to red (stiff) via simple gradient
    r = x
    g = 0.2 * (1.0 - x)
    b = 1.0 - x
    a = np.ones_like(r)

    colors = np.stack([r, g, b, a], axis=-1)
    colors = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
    return colors


def log_points_and_springs(
    object_positions: np.ndarray,
    controller_positions: Optional[np.ndarray],
    springs: np.ndarray,
    stiffness: Optional[np.ndarray],
    frame_idx: int,
    timeline: str = "frame",
) -> None:
    """Log one frame of nodes, controls, and springs to Rerun.

    This function assumes `rr.init` has already been called by the caller.
    It is designed to be testable with pure numpy inputs.
    """
    # Rerun 0.30 uses the unified set_time API instead of set_time_sequence.
    rr.set_time(timeline, sequence=int(frame_idx))

    # Nodes (object vertices)
    rr.log(
        "world/nodes",
        rr.Points3D(
            positions=object_positions,
        ),
    )

    # Control points, if any (larger radii so they're easy to spot)
    if controller_positions is not None and controller_positions.size > 0:
        rr.log(
            "world/controls",
            rr.Points3D(
                positions=controller_positions,
                radii=0.003,
                colors=[0, 255, 0, 255],
            ),
        )

    # Springs as 3D line strips — split by type for visual differentiation
    if springs.size > 0:
        num_obj = object_positions.shape[0]

        strips = build_spring_strips(
            object_positions=object_positions,
            controller_positions=controller_positions,
            springs=springs,
            num_object_points=num_obj,
        )

        obj_obj = []
        ctrl_obj = []
        ctrl_ctrl = []
        stiff_obj_obj = []
        stiff_ctrl_obj = []
        stiff_ctrl_ctrl = []

        for i in range(springs.shape[0]):
            a, b = int(springs[i, 0]), int(springs[i, 1])
            in_obj = lambda idx: idx < num_obj
            a_obj = in_obj(a)
            b_obj = in_obj(b)
            if a_obj and b_obj:
                obj_obj.append(i)
                if stiffness is not None:
                    stiff_obj_obj.append(stiffness[i])
            elif (a_obj and not b_obj) or (not a_obj and b_obj):
                ctrl_obj.append(i)
                if stiffness is not None:
                    stiff_ctrl_obj.append(stiffness[i])
            else:
                ctrl_ctrl.append(i)
                if stiffness is not None:
                    stiff_ctrl_ctrl.append(stiffness[i])

        def _log_springs(path: str, idxs: list, color_override: Optional[tuple] = None) -> None:
            if not idxs:
                return
            s = [strips[i] for i in idxs]
            if color_override is not None:
                col = np.array([color_override] * len(s), dtype=np.uint8)
            elif path == "world/springs/object_object" and stiff_obj_obj:
                col = compute_spring_colors_from_stiffness(np.array(stiff_obj_obj))
            else:
                col = None
            rr.log(path, rr.LineStrips3D(strips=s, colors=col))

        # Object–object: stiffness colors (blue → red)
        _log_springs("world/springs/object_object", obj_obj)
        # Control–object: bright yellow/cyan so they stand out
        _log_springs(
            "world/springs/control_object",
            ctrl_obj,
            color_override=(255, 220, 0, 255),
        )
        # Control–control: orange
        _log_springs(
            "world/springs/control_control",
            ctrl_ctrl,
            color_override=(255, 140, 0, 255),
        )

        # Legend (once)
        global _SPRING_LEGEND_LOGGED
        if not _SPRING_LEGEND_LOGGED:
            _SPRING_LEGEND_LOGGED = True
            rr.log(
                "world/legend",
                rr.TextLog(
                    "Springs: object-object = stiffness (BLUE soft → RED stiff). "
                    "Control-object = YELLOW. Control-control = ORANGE."
                ),
            )


def log_spring_mass_frame(
    simulator,
    frame_idx: int,
    controller_positions: Optional[torch.Tensor] = None,
    timeline: str = "frame",
) -> None:
    """Convenience wrapper: extract data from SpringMassSystemWarp and log to Rerun.

    Args:
        simulator: Instance of SpringMassSystemWarp (or an object with the same fields).
        frame_idx: Integer frame index for the Rerun timeline.
        controller_positions: Optional torch tensor of shape (num_control_points, 3)
            in world coordinates. If None, controller points will be omitted.
        timeline: Name of the Rerun time timeline (default: \"frame\").
    """
    # Extract object positions from the last substep
    object_positions = _to_numpy_vec3(simulator.wp_states[-1].wp_x, requires_grad=False)

    # Springs topology
    springs_torch = wp.to_torch(simulator.wp_springs, requires_grad=False)
    springs = springs_torch.detach().cpu().numpy()

    # Stiffness in linear space (exp of stored log-stiffness)
    spring_Y_torch = wp.to_torch(simulator.wp_spring_Y, requires_grad=False)
    spring_Y = spring_Y_torch.detach().cpu().numpy()
    stiffness = np.exp(spring_Y).astype(np.float32)

    ctrl_np: Optional[np.ndarray]
    if controller_positions is not None:
        assert controller_positions.ndim == 2 and controller_positions.shape[1] == 3
        ctrl_np = controller_positions.detach().cpu().numpy()
    else:
        ctrl_np = None

    log_points_and_springs(
        object_positions=object_positions,
        controller_positions=ctrl_np,
        springs=springs,
        stiffness=stiffness,
        frame_idx=frame_idx,
        timeline=timeline,
    )

