"""Core replay logic: load case, optional global color pre-pass, stream to Rerun."""

from __future__ import annotations

import glob
import logging
import re
import time
import warnings
from contextlib import contextmanager
from typing import Literal, overload

import numpy as np
import rerun as rr
import torch
import warp as wp
from tqdm import tqdm

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger
from rerun_viz.case_setup import load_camera_and_intrinsics, load_case_yaml_and_optimal
from rerun_viz.connect_instructions import JTAP_STYLE_SSH, REMOTE_LIVE_SERVE_SSH
from rerun_viz.port_util import pick_free_port, port_free
from rerun_viz.prepass_cache import (
    prepass_cache_file_path,
    save_prepass_cache,
    try_load_prepass_cache,
)
from rerun_viz.spring_mass_logging import (
    GlobalColorRanges,
    SpringMassLoggingOptions,
    compute_global_color_ranges,
    compute_stretch_ratios,
    log_spring_mass_frame,
    object_object_spring_row_indices,
)
from rerun_viz.terminal_output import (
    print_downsampled_bundle_summary,
    print_file_recording,
    print_performance_summary,
    print_prepass_cache_hit,
    print_replay_banner,
    print_serve_ready,
    print_ssh_hint_block,
    print_warning_line,
)


def rerun_replay_application_id(
    case_name: str | None,
    downsample_version: str | None,
) -> str:
    """
    Rerun ``application_id`` (app / blueprint name in the viewer).

    Matches default .rrd stem ``replay_<case>`` when not downsampled; with a tag appends
    ``_<tag>`` (e.g. ``replay_single_lift_sloth_kmeans_r2``).
    """
    case = (case_name or "spring_mass").strip() or "spring_mass"
    safe_case = re.sub(r"[^\w\-.]", "_", case)
    if downsample_version and (tag := downsample_version.strip()):
        safe_tag = re.sub(r"[^\w\-.]", "_", tag)
        return f"replay_{safe_case}_{safe_tag}"
    return f"replay_{safe_case}"


def set_all_seeds(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config_and_camera(base_path: str, case_name: str) -> None:
    """Configure ``cfg`` for a case (yaml, optimal params, camera)."""
    load_case_yaml_and_optimal(case_name)
    load_camera_and_intrinsics(base_path, case_name)


@contextmanager
def quiet_qqtt_stream_logs():
    """Temporarily raise qqtt stream log level so INFO lines don't spam the TTY during load."""
    h = logger.stearmhandler
    prev_log = logger.level
    prev_h = h.level
    logger.setLevel(logging.WARNING)
    h.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(prev_log)
        h.setLevel(prev_h)


@overload
def load_trainer_and_model(
    base_path: str,
    case_name: str,
    *,
    return_checkpoint_path: Literal[False] = False,
    downsample_version: str | None = None,
    emit_downsample_tty_summary: bool = False,
) -> InvPhyTrainerWarp: ...


@overload
def load_trainer_and_model(
    base_path: str,
    case_name: str,
    *,
    return_checkpoint_path: Literal[True],
    downsample_version: str | None = None,
    emit_downsample_tty_summary: bool = False,
) -> tuple[InvPhyTrainerWarp, str]: ...


def _apply_spring_mass_checkpoint(trainer: InvPhyTrainerWarp, model_path: str) -> None:
    logger.debug(f"[RERUN-REPLAY] Loading model from: {model_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint = torch.load(model_path, map_location=cfg.device)

    spring_Y = checkpoint["spring_Y"]
    collide_elas = checkpoint["collide_elas"]
    collide_fric = checkpoint["collide_fric"]
    collide_object_elas = checkpoint["collide_object_elas"]
    collide_object_fric = checkpoint["collide_object_fric"]

    assert len(spring_Y) == trainer.simulator.n_springs, (
        "Checkpoint springs do not match simulator springs"
    )

    trainer.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
    trainer.simulator.set_collide(collide_elas.detach().clone(), collide_fric.detach().clone())
    trainer.simulator.set_collide_object(
        collide_object_elas.detach().clone(),
        collide_object_fric.detach().clone(),
    )

    trainer.simulator.set_init_state(
        trainer.simulator.wp_init_vertices,
        trainer.simulator.wp_init_velocities,
        pure_inference=True,
    )


def load_trainer_and_model(
    base_path: str,
    case_name: str,
    *,
    return_checkpoint_path: bool = False,
    downsample_version: str | None = None,
    emit_downsample_tty_summary: bool = False,
) -> InvPhyTrainerWarp | tuple[InvPhyTrainerWarp, str]:
    """Create an InvPhyTrainerWarp and load spring-mass params (full or downsampled bundle)."""
    base_dir = f"experiments/{case_name}"
    logger.set_log_file(path=base_dir, name="rerun_replay_log")

    if downsample_version is not None:
        from downsampling.io import artifact_paths, load_coarse_npz
        from downsampling.validate import validate_downsampled_bundle

        info = validate_downsampled_bundle(base_path, case_name, downsample_version)
        paths = artifact_paths(base_path, case_name, downsample_version)
        z = load_coarse_npz(paths["coarse_npz"])
        precomputed_graph = {
            "init_vertices": z["init_vertices"],
            "init_springs": z["init_springs"],
            "init_rest_lengths": z["init_rest_lengths"],
            "init_masses": z["init_masses"],
            "num_object_springs": int(np.asarray(z["num_object_springs"]).item()),
        }
        trainer = InvPhyTrainerWarp(
            data_path=paths["final_data"],
            base_dir=base_dir,
            pure_inference_mode=True,
            precomputed_graph=precomputed_graph,
        )
        model_path = paths["checkpoint"]
        logger.info(
            f"[RERUN-REPLAY] Downsampled tag={downsample_version} K={info['K']} "
            f"n_springs={trainer.simulator.n_springs} dir={info['dir']}"
        )
        _apply_spring_mass_checkpoint(trainer, model_path)
        if emit_downsample_tty_summary:
            print_downsampled_bundle_summary(
                tag=downsample_version,
                K=int(info["K"]),
                n_springs=int(trainer.simulator.n_springs),
                bundle_dir=info["dir"],
                final_data_path=paths["final_data"],
                coarse_npz_path=paths["coarse_npz"],
                checkpoint_path=model_path,
            )
        if return_checkpoint_path:
            return trainer, model_path
        return trainer

    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    candidates = sorted(glob.glob(f"{base_dir}/train/best_*.pth"))
    assert len(candidates) > 0, (
        f"No best_*.pth checkpoint found under {base_dir}/train; did you run training?"
    )
    model_path = candidates[0]
    _apply_spring_mass_checkpoint(trainer, model_path)

    if return_checkpoint_path:
        return trainer, model_path
    return trainer


def collect_global_color_ranges(
    trainer: InvPhyTrainerWarp,
    start_frame: int,
    end_frame: int,
    *,
    case_name: str | None = None,
    checkpoint_path: str | None = None,
    prepass_cache: bool = True,
    prepass_refresh: bool = False,
) -> GlobalColorRanges:
    """Pre-pass: run simulator and collect stiffness, stretch, mass for across-time normalization."""
    sim = trainer.simulator
    controller_points = sim.controller_points
    assert controller_points is not None

    if prepass_cache and not prepass_refresh and case_name and checkpoint_path:
        cached = try_load_prepass_cache(case_name, start_frame, end_frame, checkpoint_path)
        if cached is not None:
            print_prepass_cache_hit(str(prepass_cache_file_path(case_name, start_frame, end_frame)))
            return cached

    springs_t = wp.to_torch(sim.wp_springs, requires_grad=False).detach().cpu().numpy()
    rest_lengths = wp.to_torch(sim.wp_rest_lengths, requires_grad=False).detach().cpu().numpy()
    spring_Y = wp.to_torch(sim.wp_spring_Y, requires_grad=False).detach().cpu().numpy()
    stiffness = np.exp(spring_Y).astype(np.float32)
    masses = (
        wp.to_torch(sim.wp_masses, requires_grad=False).detach().cpu().numpy().astype(np.float32)
    )

    stretch_list: list[np.ndarray] = []

    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities, pure_inference=True)
    for frame_idx in tqdm(
        range(start_frame, end_frame + 1),
        desc="pre-pass",
        unit="fr",
        ncols=88,
        leave=True,
    ):
        sim.set_controller_target(frame_idx, pure_inference=True)
        if sim.object_collision_flag:
            sim.update_collision_graph()
        if cfg.use_graph and hasattr(sim, "forward_graph"):
            wp.capture_launch(sim.forward_graph)
        else:
            sim.step()
        obj_pos = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).detach().cpu().numpy()
        ctrl_pos = controller_points[frame_idx].detach().cpu().numpy()
        ratios = compute_stretch_ratios(obj_pos, ctrl_pos, springs_t, rest_lengths)
        num_obj = obj_pos.shape[0]
        idx = object_object_spring_row_indices(springs_t, num_obj)
        stretch_list.append(ratios[idx] if idx.size else np.array([], dtype=np.float32))
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)

    stretch_all = (
        np.concatenate(stretch_list, axis=0) if stretch_list else np.array([], dtype=np.float32)
    )
    gr = compute_global_color_ranges(stiffness, stretch_all, masses)
    if prepass_cache and case_name and checkpoint_path:
        saved = save_prepass_cache(case_name, start_frame, end_frame, checkpoint_path, gr)
        if saved is not None:
            logger.debug(f"[RERUN-REPLAY] Wrote pre-pass cache: {saved}")
    return gr


def replay_with_rerun(
    trainer: InvPhyTrainerWarp,
    start_frame: int = 1,
    max_frames: int | None = None,
    *,
    rerun_mode: str = "file",
    grpc_port: int = 9876,
    auto_grpc_port: bool = False,
    strict_grpc_port: bool = False,
    connect_url: str | None = None,
    output_rrd: str | None = None,
    case_name: str | None = None,
    logging_options: SpringMassLoggingOptions | None = None,
    global_normalization: bool = True,
    checkpoint_path: str | None = None,
    prepass_cache: bool = True,
    prepass_refresh: bool = False,
    downsample_version: str | None = None,
    rerun_application_id: str | None = None,
) -> None:
    """Replay recorded controller_points and stream to Rerun (file, serve, or connect)."""
    simulator = trainer.simulator
    controller_points = simulator.controller_points
    assert controller_points is not None, "No controller_points found in simulator."

    total_frames = controller_points.shape[0]
    if max_frames is None:
        end_frame = total_frames - 1
    else:
        end_frame = min(total_frames - 1, start_frame + max_frames - 1)

    print_replay_banner(case_name, rerun_mode)

    opts = logging_options or SpringMassLoggingOptions()
    if global_normalization:
        gr = collect_global_color_ranges(
            trainer,
            start_frame,
            end_frame,
            case_name=case_name,
            checkpoint_path=checkpoint_path,
            prepass_cache=prepass_cache,
            prepass_refresh=prepass_refresh,
        )
        opts = SpringMassLoggingOptions(
            velocities=opts.velocities,
            forces=opts.forces,
            spring_stretch=opts.spring_stretch,
            masses=opts.masses,
            collisions=opts.collisions,
            control_interpolation=opts.control_interpolation,
            ground_plane=opts.ground_plane,
            velocity_scale=opts.velocity_scale,
            force_scale=opts.force_scale,
            global_ranges=gr,
        )
        logger.debug(f"[RERUN-REPLAY] Global ranges: {gr}")

    logger.debug(
        f"[RERUN-REPLAY] Replaying frames {start_frame}..{end_frame} "
        f"(total available: {total_frames})"
    )

    app_id = rerun_application_id or rerun_replay_application_id(case_name, downsample_version)
    rr.init(app_id, spawn=False)

    if rerun_mode == "file":
        path = output_rrd or f"{app_id}.rrd"
        rr.save(path)
        logger.debug(f"[RERUN-REPLAY] Writing to file: {path}")
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
                    f"[RERUN-REPLAY] Port {grpc_port} is in use. Free it or use --auto-grpc-port."
                )
                raise SystemExit(1)
            actual_port = grpc_port
        else:
            actual_port = grpc_port
        print_ssh_hint_block(REMOTE_LIVE_SERVE_SSH)
        uri = rr.serve_grpc(grpc_port=actual_port)
        logger.debug(f"[RERUN-REPLAY] Serving Rerun gRPC at: {uri}")
        print_serve_ready(actual_port, str(uri))
    elif rerun_mode == "connect":
        print_ssh_hint_block(JTAP_STYLE_SSH)
        try:
            rr.connect_grpc(url=connect_url)
        except Exception as exc:
            logger.error(
                f"[RERUN-REPLAY] connect_grpc failed ({exc}). "
                "Is the viewer running? Is SSH -R 9876:127.0.0.1:9876 set?"
            )
            raise
    else:
        raise ValueError(f"Unknown --rerun_mode: {rerun_mode} (use file, serve, or connect)")

    rr.log("/", rr.Clear(recursive=True))

    simulator.set_init_state(
        simulator.wp_init_vertices,
        simulator.wp_init_velocities,
        pure_inference=True,
    )

    num_frames = end_frame - start_frame + 1
    n_vertices = simulator.n_vertices
    n_springs = simulator.n_springs
    num_substeps = simulator.num_substeps

    frame_range = range(start_frame, end_frame + 1)
    t_physics_total = 0.0
    for frame_idx in tqdm(frame_range, desc="replay", unit="fr", ncols=88):
        simulator.set_controller_target(frame_idx, pure_inference=True)

        if simulator.object_collision_flag:
            simulator.update_collision_graph()

        t_phys_start = time.perf_counter()
        if cfg.use_graph and hasattr(simulator, "forward_graph"):
            wp.capture_launch(simulator.forward_graph)
        else:
            simulator.step()
        t_physics_total += time.perf_counter() - t_phys_start

        current_ctrl = controller_points[frame_idx]
        log_spring_mass_frame(
            simulator=simulator,
            frame_idx=frame_idx,
            controller_positions=current_ctrl,
            timeline="frame",
            options=opts,
        )

        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
            pure_inference=True,
        )

    print_performance_summary(
        n_vertices=n_vertices,
        n_springs=n_springs,
        num_substeps=num_substeps,
        num_frames=num_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        t_physics_total=t_physics_total,
    )

    logger.debug("[RERUN-REPLAY] Finished streaming to Rerun.")
