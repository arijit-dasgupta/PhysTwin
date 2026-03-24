"""Load full-res + downsampled trainers; replay shared controls + per-variant spring meshes."""

from __future__ import annotations

import colorsys
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import warp as wp
from tqdm import tqdm

from downsampling.validate import downsampled_bundle_exists
from qqtt.utils import cfg, logger
from rerun_viz.compare_logging import (
    log_compare_legend_once,
    log_mesh_variant_frame,
    log_shared_controls_frame,
    mesh_entity_path,
)
from rerun_viz.connect_instructions import JTAP_STYLE_SSH, REMOTE_LIVE_SERVE_SSH
from rerun_viz.port_util import pick_free_port, port_free
from rerun_viz.replay_core import (
    load_config_and_camera,
    load_trainer_and_model,
    print_file_recording,
    print_serve_ready,
    print_ssh_hint_block,
    print_warning_line,
    quiet_qqtt_stream_logs,
    set_all_seeds,
)
from rerun_viz.spring_mass_logging import _to_numpy_vec3
from rerun_viz.surface_mesh import (
    SurfaceMeshError,
    build_surface_mesh,
    subsample_triangles_to_budget,
)
from rerun_viz.terminal_output import (
    print_multi_mesh_precompute_footer,
    print_multi_mesh_precompute_header,
)


def rerun_multi_application_id(case_name: str | None) -> str:
    case = (case_name or "spring_mass").strip() or "spring_mass"
    safe = re.sub(r"[^\w\-.]", "_", case)
    return f"replay_multi_{safe}"


def list_auto_downsample_tags(base_path: str, case_name: str) -> list[str]:
    root = Path(base_path) / case_name / "downsampled"
    if not root.is_dir():
        return []
    tags: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and downsampled_bundle_exists(base_path, case_name, child.name):
            tags.append(child.name)
    return tags


def _springs_numpy(sim) -> np.ndarray:
    t = wp.to_torch(sim.wp_springs, requires_grad=False).detach().cpu().numpy()
    return np.asarray(t, dtype=np.int64)


def _assert_controllers_compatible(trainers: list) -> None:
    ref = trainers[0].controller_points
    assert ref is not None
    for tr in trainers[1:]:
        c = tr.controller_points
        assert c is not None
        assert c.shape == ref.shape, "controller_points shape mismatch between variants"
    for tr in trainers[1:]:
        c = tr.controller_points
        assert c is not None
        if not torch.allclose(c, ref, atol=5e-3, rtol=0):
            logger.warning(
                "[replay_multi] controller_points differ slightly from full_res; "
                "each simulator still uses its own loaded trajectory."
            )
            break


@dataclass
class _VariantMesh:
    label: str
    trainer: object
    n_mesh: int
    triangle_indices: np.ndarray
    rgb: tuple[int, int, int]
    opacity: float


def _distinct_srgb_colors(n: int) -> list[tuple[int, int, int]]:
    """Return ``n`` distinct saturated sRGB colors (0–255).

    Hues are **evenly spaced** on the wheel so every variant gets a different color, including
    ``full_res`` (index ``0``) vs every downsampled tag. Works for any ``n`` (no wrap/repeat).
    """
    if n <= 0:
        return []
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        # Spread samples across hue; avoid pure black/white via fixed S and V
        h = (i + 0.5) / float(n)
        r, g, b = colorsys.hsv_to_rgb(h, 0.88, 0.97)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def precompute_variant_meshes(
    *,
    labels: list[str],
    trainers: list,
    mesh_use_original_only: bool,
    mesh_spring_max_edge_factor: float,
    mesh_opacity: float,
    mesh_max_triangles: int,
) -> list[_VariantMesh]:
    print_multi_mesh_precompute_header()
    out: list[_VariantMesh] = []
    mesh_ns: list[int] = []
    n_tris: list[int] = []

    specs = list(zip(labels, trainers, strict=True))
    color_palette = _distinct_srgb_colors(len(specs))
    pbar = tqdm(
        specs,
        desc="mesh build",
        unit="model",
        ncols=100,
        leave=True,
    )
    for idx, (label, trainer) in enumerate(pbar):
        n_mesh = (
            int(trainer.num_original_points)
            if mesh_use_original_only
            else int(trainer.num_surface_points)
        )
        rest = trainer.structure_points.detach().cpu().numpy()
        springs = _springs_numpy(trainer.simulator)
        try:
            _verts, tri = build_surface_mesh(
                rest,
                springs,
                n_mesh,
                spring_max_edge_factor=mesh_spring_max_edge_factor,
            )
        except SurfaceMeshError as e:
            raise RuntimeError(f"{label}: spring mesh failed: {e}") from e
        tri = subsample_triangles_to_budget(rest[:n_mesh], tri, mesh_max_triangles)
        rgb = color_palette[idx]
        out.append(
            _VariantMesh(
                label=label,
                trainer=trainer,
                n_mesh=n_mesh,
                triangle_indices=tri,
                rgb=rgb,
                opacity=mesh_opacity,
            )
        )
        nv = int(n_mesh)
        nt = int(tri.shape[0])
        mesh_ns.append(nv)
        n_tris.append(nt)
        pbar.set_postfix_str(f"{label}  tris={nt:,}", refresh=True)

    print_multi_mesh_precompute_footer(labels=labels, mesh_ns=mesh_ns, n_tris=n_tris)
    return out


def replay_multiple_models_with_rerun(
    base_path: str,
    case_name: str,
    *,
    tags: Sequence[str] | None,
    start_frame: int = 1,
    max_frames: int | None = None,
    rerun_mode: str = "file",
    grpc_port: int = 9876,
    auto_grpc_port: bool = False,
    strict_grpc_port: bool = False,
    connect_url: str | None = None,
    output_rrd: str | None = None,
    mesh_use_original_only: bool = False,
    mesh_spring_max_edge_factor: float = 6.0,
    mesh_max_triangles: int = 0,
    mesh_opacity: float = 0.25,
    mesh_alpha: int | None = None,
) -> None:
    if cfg.data_type != "real":
        raise SystemExit("[replay_multi] requires cfg.data_type == 'real' (RealData).")

    set_all_seeds(42)

    if mesh_alpha is not None:
        opacity = float(np.clip(mesh_alpha / 255.0, 0.0, 1.0))
    else:
        opacity = float(np.clip(mesh_opacity, 0.0, 1.0))

    if tags is None:
        tag_list = list_auto_downsample_tags(base_path, case_name)
    else:
        tag_list = list(tags)

    labels: list[str] = ["full_res"]
    with quiet_qqtt_stream_logs():
        load_config_and_camera(base_path, case_name)
        trainers_list: list = []
        tr0 = load_trainer_and_model(base_path, case_name, downsample_version=None)
        trainers_list.append(tr0)
        for tag in tag_list:
            tr = load_trainer_and_model(
                base_path,
                case_name,
                downsample_version=tag,
                emit_downsample_tty_summary=False,
            )
            trainers_list.append(tr)
            labels.append(tag)

    _assert_controllers_compatible(trainers_list)

    mesh_variants = precompute_variant_meshes(
        labels=labels,
        trainers=trainers_list,
        mesh_use_original_only=mesh_use_original_only,
        mesh_spring_max_edge_factor=mesh_spring_max_edge_factor,
        mesh_opacity=opacity,
        mesh_max_triangles=mesh_max_triangles,
    )

    ref = trainers_list[0]
    controller_points = ref.simulator.controller_points
    assert controller_points is not None
    total_frames = int(controller_points.shape[0])
    end_frame = total_frames - 1 if max_frames is None else min(total_frames - 1, start_frame + max_frames - 1)

    app_id = rerun_multi_application_id(case_name)
    rr.init(app_id, spawn=False)

    if rerun_mode == "file":
        path = output_rrd or f"{app_id}.rrd"
        rr.save(path)
        logger.debug(f"[RERUN-MULTI] Writing to file: {path}")
        print_file_recording(path)
    elif rerun_mode == "serve":
        if strict_grpc_port and auto_grpc_port:
            logger.error("--strict-grpc-port and --auto-grpc-port are mutually exclusive")
            raise SystemExit(1)
        if auto_grpc_port:
            actual_port, used_fallback = pick_free_port(grpc_port)
            if used_fallback:
                print_warning_line(f"port {grpc_port} busy → using {actual_port} (match SSH + URL)")
        elif strict_grpc_port:
            if not port_free(grpc_port):
                logger.error(
                    f"[RERUN-MULTI] Port {grpc_port} is in use. Free it or use --auto-grpc-port."
                )
                raise SystemExit(1)
            actual_port = grpc_port
        else:
            actual_port = grpc_port
        print_ssh_hint_block(REMOTE_LIVE_SERVE_SSH)
        uri = rr.serve_grpc(grpc_port=actual_port)
        logger.debug(f"[RERUN-MULTI] Serving Rerun gRPC at: {uri}")
        print_serve_ready(actual_port, str(uri))
    elif rerun_mode == "connect":
        print_ssh_hint_block(JTAP_STYLE_SSH)
        try:
            rr.connect_grpc(url=connect_url)
        except Exception as exc:
            logger.error(
                f"[RERUN-MULTI] connect_grpc failed ({exc}). "
                "Is the viewer running? Is SSH -R 9876:127.0.0.1:9876 set?"
            )
            raise
    else:
        raise ValueError(f"Unknown --rerun_mode: {rerun_mode}")

    rr.log("/", rr.Clear(recursive=True))

    legend_lines = [
        "Multi-model compare: shared controls (yellow/orange) + one mesh per variant "
        f"(opacity ≈ {opacity:.2f} via Mesh3D albedo_factor; tune --mesh-opacity / --mesh-alpha).",
        "Per-variant paths: " + ", ".join(mesh_entity_path(m.label) for m in mesh_variants),
    ]
    log_compare_legend_once("\n".join(legend_lines))

    for tr in trainers_list:
        tr.simulator.set_init_state(
            tr.simulator.wp_init_vertices,
            tr.simulator.wp_init_velocities,
            pure_inference=True,
        )

    frame_range = range(start_frame, end_frame + 1)
    full_sim = trainers_list[0].simulator

    for frame_idx in tqdm(frame_range, desc="compare replay", unit="fr", ncols=100):
        for mv in mesh_variants:
            sim = mv.trainer.simulator
            sim.set_controller_target(frame_idx, pure_inference=True)
            if sim.object_collision_flag:
                sim.update_collision_graph()
            if cfg.use_graph and hasattr(sim, "forward_graph"):
                wp.capture_launch(sim.forward_graph)
            else:
                sim.step()

        obj_pos = _to_numpy_vec3(full_sim.wp_states[-1].wp_x, requires_grad=False)
        ctrl = controller_points[frame_idx]
        ctrl_np = ctrl.detach().cpu().numpy()
        springs_full = _springs_numpy(full_sim)

        log_shared_controls_frame(
            object_positions=obj_pos,
            controller_positions=ctrl_np,
            springs=springs_full,
            frame_idx=frame_idx,
            timeline="frame",
        )

        for mv in mesh_variants:
            sim = mv.trainer.simulator
            pos = _to_numpy_vec3(sim.wp_states[-1].wp_x, requires_grad=False)[: mv.n_mesh]
            log_mesh_variant_frame(
                label=mv.label,
                vertex_positions=pos,
                triangle_indices=mv.triangle_indices,
                rgb=mv.rgb,
                opacity=mv.opacity,
                frame_idx=frame_idx,
                timeline="frame",
            )

        for mv in mesh_variants:
            sim = mv.trainer.simulator
            sim.set_init_state(
                sim.wp_states[-1].wp_x,
                sim.wp_states[-1].wp_v,
                pure_inference=True,
            )

    logger.debug("[RERUN-MULTI] Finished streaming to Rerun.")
