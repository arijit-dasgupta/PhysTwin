from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_SPRING_LEGEND_LOGGED = False
_PHYSICS_LEGEND_LOGGED = False
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


def compute_stretch_ratios(
    object_positions: np.ndarray,
    controller_positions: Optional[np.ndarray],
    springs: np.ndarray,
    rest_lengths: np.ndarray,
) -> np.ndarray:
    """Compute current_len/rest for each spring. Returns (n_springs,) float32."""
    # Validate inputs: NaN here would propagate and cause black colors
    if np.any(~np.isfinite(object_positions)):
        raise ValueError(
            "object_positions contains NaN/inf; simulator state may be invalid"
        )
    if controller_positions is not None and np.any(~np.isfinite(controller_positions)):
        raise ValueError(
            "controller_positions contains NaN/inf"
        )
    if np.any(~np.isfinite(rest_lengths)):
        raise ValueError(
            "rest_lengths contains NaN/inf; check wp_rest_lengths"
        )
    strips = build_spring_strips(
        object_positions, controller_positions, springs, object_positions.shape[0]
    )
    ratios = np.zeros(springs.shape[0], dtype=np.float32)
    for i in range(springs.shape[0]):
        rest = float(rest_lengths[i])
        if rest > 1e-8:
            current_len = float(np.linalg.norm(strips[i, 1] - strips[i, 0]))
            ratios[i] = current_len / rest
        else:
            ratios[i] = 1.0
    return ratios


def _normalize_percentile(
    vals: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    v_low: Optional[float] = None,
    v_high: Optional[float] = None,
) -> np.ndarray:
    """Normalize values to [0, 1]. Use v_low/v_high if provided (global), else percentile."""
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size == 0:
        return vals
    if v_low is not None and v_high is not None:
        if not np.isfinite(v_low) or not np.isfinite(v_high):
            return np.full_like(vals, 0.5)
        denom = v_high - v_low
    else:
        v_low = np.nanpercentile(vals, low_pct)
        v_high = np.nanpercentile(vals, high_pct)
        if not np.isfinite(v_low) or not np.isfinite(v_high):
            return np.full_like(vals, 0.5)
        denom = v_high - v_low
    if denom < 1e-12:
        return np.full_like(vals, 0.5)
    x = (vals - v_low) / denom
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _value_to_color(x: np.ndarray) -> np.ndarray:
    """Map [0,1] to RGBA: blue (0) -> purple (mid) -> red (1). Used for stiffness, mass."""
    x = np.asarray(x, dtype=np.float32)
    r = x
    g = 0.2 * (1.0 - x)
    b = 1.0 - x
    a = np.ones_like(r)
    colors = np.stack([r, g, b, a], axis=-1)
    return np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)


def _value_to_color_stretch(x: np.ndarray) -> np.ndarray:
    """Map [0,1] to RGBA: dark red (compressed) -> bright orange (stretched). Red-orange spectrum, light-dark contrast."""
    x = np.asarray(x, dtype=np.float32)
    if np.any(~np.isfinite(x)):
        raise ValueError(
            "Stretch normalization produced NaN/inf; check ratios and global range"
        )
    x = np.clip(x, 0.0, 1.0)
    # Dark red (100, 25, 20) -> bright orange (255, 165, 60)
    r = np.clip(100.0 + 155.0 * x, 0.0, 255.0)
    g = np.clip(25.0 + 140.0 * x, 0.0, 255.0)
    b = np.clip(20.0 + 40.0 * x, 0.0, 255.0)
    a = np.full_like(r, 255.0)
    colors = np.stack([r, g, b, a], axis=-1)
    return colors.astype(np.uint8)


def compute_spring_colors_from_stiffness(
    stiffness: np.ndarray,
    v_low: Optional[float] = None,
    v_high: Optional[float] = None,
) -> np.ndarray:
    """Map stiffness to RGBA: blue (soft) -> purple -> red (stiff)."""
    if stiffness.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    x = _normalize_percentile(stiffness, v_low=v_low, v_high=v_high)
    return _value_to_color(x)


def compute_spring_colors_from_stretch(
    ratios: np.ndarray,
    v_low: Optional[float] = None,
    v_high: Optional[float] = None,
) -> np.ndarray:
    """Map stretch ratio to RGBA: dark red (compressed) -> bright orange (stretched)."""
    if ratios.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    # Fall back to per-frame percentile when global range is too narrow (all values same)
    if v_low is not None and v_high is not None and (v_high - v_low) < 1e-6:
        v_low, v_high = None, None
    x = _normalize_percentile(ratios, v_low=v_low, v_high=v_high)
    return _value_to_color_stretch(x)


@dataclass
class GlobalColorRanges:
    """Optional fixed (v_low, v_high) for normalization across time. None = per-frame percentile."""

    stiffness: Optional[Tuple[float, float]] = None
    stretch: Optional[Tuple[float, float]] = None
    mass: Optional[Tuple[float, float]] = None


def compute_global_color_ranges(
    stiffness_all: np.ndarray,
    stretch_ratios_all: np.ndarray,
    mass_all: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> GlobalColorRanges:
    """Compute (v_low, v_high) percentiles from arrays collected across all frames."""
    def _range(arr: np.ndarray) -> Optional[Tuple[float, float]]:
        a = np.asarray(arr, dtype=np.float32).flatten()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return None
        lo = float(np.nanpercentile(a, low_pct))
        hi = float(np.nanpercentile(a, high_pct))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        return (lo, hi)

    return GlobalColorRanges(
        stiffness=_range(stiffness_all),
        stretch=_range(stretch_ratios_all),
        mass=_range(mass_all),
    )


@dataclass
class SpringMassLoggingOptions:
    """Options to control which physics views are logged and their scaling."""

    velocities: bool = True
    forces: bool = True
    spring_stretch: bool = True
    masses: bool = True
    collisions: bool = True
    control_interpolation: bool = True
    ground_plane: bool = True
    velocity_scale: float = 0.05
    force_scale: float = 1e-4
    global_ranges: Optional[GlobalColorRanges] = None


def log_velocities(
    positions: np.ndarray,
    velocities: np.ndarray,
    scale: float,
) -> None:
    """Log velocity arrows to physics/velocities."""
    if positions.size == 0 or velocities.size == 0:
        return
    vectors = (velocities * scale).astype(np.float32)
    rr.log("physics/velocities", rr.Arrows3D(origins=positions, vectors=vectors))


def log_forces(
    positions: np.ndarray,
    forces: np.ndarray,
    scale: float,
) -> None:
    """Log force arrows to physics/forces."""
    if positions.size == 0 or forces.size == 0:
        return
    vectors = (forces * scale).astype(np.float32)
    rr.log("physics/forces", rr.Arrows3D(origins=positions, vectors=vectors))


def log_springs_by_stretch(
    object_positions: np.ndarray,
    controller_positions: Optional[np.ndarray],
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    stretch_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Log springs colored by stretch ratio to physics/springs_stretch. Dark red (compressed) -> bright orange (stretched).
    Only object-object springs (both endpoints are object vertices); no control-related springs."""
    if springs.size == 0:
        return
    if np.any(~np.isfinite(object_positions)) or np.any(~np.isfinite(rest_lengths)):
        raise ValueError(
            "object_positions or rest_lengths contains NaN/inf; simulator state may be invalid"
        )
    num_obj = object_positions.shape[0]
    in_obj = lambda idx: idx < num_obj
    # Only object-object: both endpoints must be object vertices
    keep = [
        i for i in range(springs.shape[0])
        if in_obj(int(springs[i, 0])) and in_obj(int(springs[i, 1]))
    ]
    if not keep:
        return
    springs_sub = springs[keep]
    rest_sub = rest_lengths[keep]
    strips = build_spring_strips(
        object_positions=object_positions,
        controller_positions=controller_positions,
        springs=springs_sub,
        num_object_points=num_obj,
    )
    ratios = np.zeros(len(keep), dtype=np.float32)
    for j, i in enumerate(keep):
        rest = float(rest_lengths[i])
        if rest > 1e-8:
            current_len = float(np.linalg.norm(strips[j, 1] - strips[j, 0]))
            ratios[j] = current_len / rest
        else:
            ratios[j] = 1.0
    v_low, v_high = stretch_range if stretch_range else (None, None)
    colors = compute_spring_colors_from_stretch(ratios, v_low=v_low, v_high=v_high)
    strips_list = [strips[j] for j in range(len(strips))]
    # Same format as stiffness springs: RGBA, list of strips
    rr.log(
        "physics/springs_stretch",
        rr.LineStrips3D(strips=strips_list, colors=colors),
    )


def log_masses(
    positions: np.ndarray,
    masses: np.ndarray,
    min_radius: float = 0.001,
    max_radius: float = 0.006,
    mass_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Log nodes with radii and colors proportional to mass (percentile-normalized).

    Light (low mass) = blue, small; heavy (high mass) = red, large.
    Uses 2nd–98th percentile for robustness. When mass_range provided, normalizes across time.
    """
    if positions.size == 0 or masses.size == 0:
        return
    masses_flat = masses.astype(np.float32).flatten()
    v_low, v_high = mass_range if mass_range else (None, None)
    x = _normalize_percentile(masses_flat, v_low=v_low, v_high=v_high, low_pct=2.0, high_pct=98.0)
    radii = (min_radius + x * (max_radius - min_radius)).astype(np.float32)
    colors = _value_to_color(x)
    rr.log("physics/masses", rr.Points3D(positions=positions, radii=radii, colors=colors))


def log_collisions(
    positions: np.ndarray,
    collision_indices: np.ndarray,
    collision_number: np.ndarray,
) -> None:
    """Log collision pairs as line strips to physics/collisions."""
    strips = []
    for i in range(positions.shape[0]):
        n = int(collision_number[i])
        for k in range(n):
            j = int(collision_indices[i, k])
            if i < j:
                strips.append(np.array([positions[i], positions[j]], dtype=np.float32))
    if not strips:
        return
    rr.log(
        "physics/collisions",
        rr.LineStrips3D(
            strips=strips,
            colors=np.array([[255, 0, 255, 255]] * len(strips), dtype=np.uint8),
        ),
    )


def log_control_interpolation(
    orig_control: np.ndarray,
    target_control: np.ndarray,
) -> None:
    """Log original and target control points with connecting lines to controls/interpolation."""
    if orig_control.size == 0 and target_control.size == 0:
        return
    if orig_control.size > 0:
        rr.log(
            "controls/interpolation/original",
            rr.Points3D(positions=orig_control, radii=0.004, colors=[255, 0, 0, 255]),
        )
    if target_control.size > 0:
        rr.log(
            "controls/interpolation/target",
            rr.Points3D(positions=target_control, radii=0.004, colors=[0, 0, 255, 255]),
        )
    if orig_control.size > 0 and target_control.size > 0 and orig_control.shape == target_control.shape:
        strips = [np.array([orig_control[i], target_control[i]], dtype=np.float32) for i in range(orig_control.shape[0])]
        rr.log(
            "controls/interpolation/links",
            rr.LineStrips3D(strips=strips, colors=np.array([[200, 200, 0, 200]] * len(strips), dtype=np.uint8)),
        )


def log_ground_plane(
    size: float = 15.0,
    thickness: float = 0.02,
    z: float = 0.0,
) -> None:
    """Log a visible ground plane at z. Solid fill, opaque (Rerun alpha not fully supported for boxes)."""
    rr.log(
        "world/ground",
        rr.Boxes3D(
            centers=np.array([[0.0, 0.0, z]], dtype=np.float32),
            half_sizes=np.array([[size, size, thickness]], dtype=np.float32),
            colors=[[200, 200, 210]],  # Light gray, opaque
            fill_mode="solid",
        ),
    )


def log_points_and_springs(
    object_positions: np.ndarray,
    controller_positions: Optional[np.ndarray],
    springs: np.ndarray,
    stiffness: Optional[np.ndarray],
    frame_idx: int,
    timeline: str = "frame",
    stiffness_range: Optional[Tuple[float, float]] = None,
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

        v_low, v_high = stiffness_range if stiffness_range else (None, None)

        def _log_springs(path: str, idxs: list, color_override: Optional[tuple] = None) -> None:
            if not idxs:
                return
            s = [strips[i] for i in idxs]
            if color_override is not None:
                col = np.array([color_override] * len(s), dtype=np.uint8)
            elif path == "world/springs/object_object" and stiff_obj_obj:
                col = compute_spring_colors_from_stiffness(np.array(stiff_obj_obj), v_low=v_low, v_high=v_high)
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
    options: Optional[SpringMassLoggingOptions] = None,
) -> None:
    """Convenience wrapper: extract data from SpringMassSystemWarp and log to Rerun.

    Args:
        simulator: Instance of SpringMassSystemWarp (or an object with the same fields).
        frame_idx: Integer frame index for the Rerun timeline.
        controller_positions: Optional torch tensor of shape (num_control_points, 3)
            in world coordinates. If None, controller points will be omitted.
        timeline: Name of the Rerun time timeline (default: \"frame\").
        options: Optional logging options. If None, defaults (all views on) are used.
    """
    opts = options or SpringMassLoggingOptions()

    # Rerun 0.30 uses the unified set_time API
    rr.set_time(timeline, sequence=int(frame_idx))

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

    gr = opts.global_ranges
    stiffness_range = gr.stiffness if gr else None
    log_points_and_springs(
        object_positions=object_positions,
        controller_positions=ctrl_np,
        springs=springs,
        stiffness=stiffness,
        frame_idx=frame_idx,
        timeline=timeline,
        stiffness_range=stiffness_range,
    )

    # Physics views
    if opts.velocities:
        velocities = _to_numpy_vec3(simulator.wp_states[-1].wp_v, requires_grad=False)
        log_velocities(object_positions, velocities, opts.velocity_scale)
    if opts.forces:
        forces = _to_numpy_vec3(simulator.wp_states[-1].wp_vertice_forces, requires_grad=False)
        log_forces(object_positions, forces, opts.force_scale)
    if opts.spring_stretch:
        rest_lengths = wp.to_torch(simulator.wp_rest_lengths, requires_grad=False).detach().cpu().numpy()
        stretch_range = gr.stretch if gr else None
        log_springs_by_stretch(object_positions, ctrl_np, springs, rest_lengths, stretch_range=stretch_range)
    if opts.masses:
        masses = wp.to_torch(simulator.wp_masses, requires_grad=False).detach().cpu().numpy()
        mass_range = gr.mass if gr else None
        log_masses(object_positions, masses, mass_range=mass_range)
    if opts.collisions and getattr(simulator, "object_collision_flag", 0):
        coll_idx = wp.to_torch(simulator.wp_collision_indices, requires_grad=False).detach().cpu().numpy()
        coll_num = wp.to_torch(simulator.wp_collision_number, requires_grad=False).detach().cpu().numpy()
        log_collisions(object_positions, coll_idx, coll_num)
    if opts.control_interpolation and ctrl_np is not None and simulator.num_control_points > 0:
        orig = _to_numpy_vec3(simulator.wp_original_control_point, requires_grad=False)
        tgt = _to_numpy_vec3(simulator.wp_target_control_point, requires_grad=False)
        log_control_interpolation(orig, tgt)
    if opts.ground_plane:
        log_ground_plane()

    # Legend for new views (once)
    global _PHYSICS_LEGEND_LOGGED
    if not _PHYSICS_LEGEND_LOGGED:
        _PHYSICS_LEGEND_LOGGED = True
        rr.log(
            "physics/legend",
            rr.TextLog(
                "physics/velocities: velocity arrows. physics/forces: force arrows. "
                "physics/springs_stretch: stretch ratio DARK RED=compressed -> BRIGHT ORANGE=stretched (percentile). "
                "physics/masses: BLUE=light/small, RED=heavy/large (percentile). "
                "physics/collisions: collision pairs. controls/interpolation: original vs target."
            ),
        )

