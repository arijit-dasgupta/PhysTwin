from __future__ import annotations

import glob
import json
import os
import sys
import time
from argparse import ArgumentParser
from typing import List, Optional

# Ensure project root is on sys.path so qqtt is importable when run from anywhere
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import rerun as rr
import torch
import warp as wp
from tqdm import tqdm

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger
from rerun_viz.connect_instructions import JTAP_STYLE_SSH, REMOTE_LIVE_SERVE_SSH
from rerun_viz.port_util import pick_free_port, port_free
from rerun_viz.spring_mass_logging import (
    GlobalColorRanges,
    SpringMassLoggingOptions,
    compute_global_color_ranges,
    compute_stretch_ratios,
    log_spring_mass_frame,
)


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
    """Mirror the inference/interactive scripts to configure cfg for a case."""
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    # Read first-stage optimized parameters
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    import pickle

    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Camera intrinsics/extrinsics are not strictly required for physics-only
    # replay, but we keep them consistent with other scripts.
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]


def load_trainer_and_model(
    base_path: str,
    case_name: str,
) -> InvPhyTrainerWarp:
    """Create an InvPhyTrainerWarp and load the best trained spring-mass params."""
    base_dir = f"experiments/{case_name}"
    logger.set_log_file(path=base_dir, name="rerun_replay_log")

    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    # Load best model checkpoint (same pattern as other scripts)
    candidates = glob.glob(f"{base_dir}/train/best_*.pth")
    assert (
        len(candidates) > 0
    ), f"No best_*.pth checkpoint found under {base_dir}/train; did you run training?"
    model_path = candidates[0]
    logger.info(f"[RERUN-REPLAY] Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=cfg.device)

    spring_Y = checkpoint["spring_Y"]
    collide_elas = checkpoint["collide_elas"]
    collide_fric = checkpoint["collide_fric"]
    collide_object_elas = checkpoint["collide_object_elas"]
    collide_object_fric = checkpoint["collide_object_fric"]

    assert (
        len(spring_Y) == trainer.simulator.n_springs
    ), "Checkpoint springs do not match simulator springs"

    trainer.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
    trainer.simulator.set_collide(
        collide_elas.detach().clone(), collide_fric.detach().clone()
    )
    trainer.simulator.set_collide_object(
        collide_object_elas.detach().clone(),
        collide_object_fric.detach().clone(),
    )

    # Initialize simulator state at rest
    trainer.simulator.set_init_state(
        trainer.simulator.wp_init_vertices,
        trainer.simulator.wp_init_velocities,
        pure_inference=True,
    )

    return trainer


def collect_global_color_ranges(
    trainer: InvPhyTrainerWarp,
    start_frame: int,
    end_frame: int,
) -> GlobalColorRanges:
    """Pre-pass: run simulator through frames and collect stiffness, stretch, mass for across-time normalization."""
    sim = trainer.simulator
    controller_points = sim.controller_points
    assert controller_points is not None

    springs_t = wp.to_torch(sim.wp_springs, requires_grad=False).detach().cpu().numpy()
    rest_lengths = wp.to_torch(sim.wp_rest_lengths, requires_grad=False).detach().cpu().numpy()
    spring_Y = wp.to_torch(sim.wp_spring_Y, requires_grad=False).detach().cpu().numpy()
    stiffness = np.exp(spring_Y).astype(np.float32)
    masses = wp.to_torch(sim.wp_masses, requires_grad=False).detach().cpu().numpy().astype(np.float32)

    stretch_list: List[np.ndarray] = []

    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities, pure_inference=True)
    for frame_idx in range(start_frame, end_frame + 1):
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
        # Only object-object springs (same filter as log_springs_by_stretch)
        num_obj = obj_pos.shape[0]
        keep = [i for i in range(springs_t.shape[0]) if int(springs_t[i, 0]) < num_obj and int(springs_t[i, 1]) < num_obj]
        stretch_list.append(ratios[keep])
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)

    stretch_all = np.concatenate(stretch_list, axis=0) if stretch_list else np.array([], dtype=np.float32)
    return compute_global_color_ranges(stiffness, stretch_all, masses)


def replay_with_rerun(
    trainer: InvPhyTrainerWarp,
    start_frame: int = 1,
    max_frames: Optional[int] = None,
    *,
    rerun_mode: str = "file",
    grpc_port: int = 9876,
    auto_grpc_port: bool = False,
    strict_grpc_port: bool = False,
    connect_url: Optional[str] = None,
    output_rrd: Optional[str] = None,
    case_name: Optional[str] = None,
    logging_options: Optional[SpringMassLoggingOptions] = None,
    global_normalization: bool = True,
) -> None:
    """Replay a sequence driven by recorded controller_points and stream to Rerun.

    **Default ``rerun_mode="file"``:** Write to an .rrd file. Copy to laptop and open with
    ``rerun path/to/output.rrd``. No network needed.

    **``rerun_mode="serve"``:** ``rr.serve_grpc`` — VM hosts; laptop uses SSH -L +
    ``rerun rerun+http://127.0.0.1:<port>/proxy``.

    **``rerun_mode="connect"`` (JTAP-style):** ``rr.connect_grpc`` — use SSH -R.
    """
    simulator = trainer.simulator
    controller_points = simulator.controller_points  # (T, num_ctrl, 3)
    assert controller_points is not None, "No controller_points found in simulator."

    total_frames = controller_points.shape[0]
    if max_frames is None:
        end_frame = total_frames - 1
    else:
        end_frame = min(total_frames - 1, start_frame + max_frames - 1)

    opts = logging_options or SpringMassLoggingOptions()
    if global_normalization:
        logger.info("[RERUN-REPLAY] Pre-pass: collecting global color ranges across time...")
        gr = collect_global_color_ranges(trainer, start_frame, end_frame)
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
        logger.info(f"[RERUN-REPLAY] Global ranges: stiffness={gr.stiffness} stretch={gr.stretch} mass={gr.mass}")

    logger.info(
        f"[RERUN-REPLAY] Replaying frames {start_frame}..{end_frame} "
        f"(total available: {total_frames})"
    )

    rr.init("phystwin_spring_mass_replay", spawn=False)

    if rerun_mode == "file":
        path = output_rrd or f"replay_{case_name or 'spring_mass'}.rrd"
        rr.save(path)
        logger.info(f"[RERUN-REPLAY] Writing to file: {path}")
        print(f"\n[RERUN] Writing to: {path}\nCopy to your laptop and run:  rerun {path}\n", flush=True)
    elif rerun_mode == "serve":
        # VM / remote: Python hosts the server; viewer on host connects via port-forward.
        # See rerun_viz/README.md
        if strict_grpc_port and auto_grpc_port:
            logger.error("--strict-grpc-port and --auto-grpc-port are mutually exclusive")
            raise SystemExit(1)
        if auto_grpc_port:
            actual_port, used_fallback = pick_free_port(grpc_port)
            if used_fallback:
                logger.warning(
                    f"[RERUN-REPLAY] Port {grpc_port} busy; serving on {actual_port} instead. "
                    "Match SSH port-forward and rerun URL to this port."
                )
        elif strict_grpc_port:
            if not port_free(grpc_port):
                logger.error(
                    f"[RERUN-REPLAY] Port {grpc_port} is in use. "
                    "Free it, or use --auto-grpc-port to scan for a free port."
                )
                raise SystemExit(1)
            actual_port = grpc_port
        else:
            # Default: use grpc_port as-is (no pre-scan; avoids false "busy" in VMs).
            actual_port = grpc_port
        print(REMOTE_LIVE_SERVE_SSH, flush=True)
        uri = rr.serve_grpc(grpc_port=actual_port)
        logger.info(f"[RERUN-REPLAY] Serving Rerun gRPC at: {uri}")
        print(
            f"\n[RERUN] Server URI: {uri}\n"
            f"On your LAPTOP (with -L {actual_port}:127.0.0.1:{actual_port} active), run:\n"
            f"  rerun rerun+http://127.0.0.1:{actual_port}/proxy\n",
            flush=True,
        )
    elif rerun_mode == "connect":
        print(JTAP_STYLE_SSH, flush=True)
        try:
            rr.connect_grpc(url=connect_url)
        except Exception as exc:
            logger.error(
                "[RERUN-REPLAY] connect_grpc failed (%s). "
                "Is the viewer running on your laptop? Is SSH -R 9876:127.0.0.1:9876 set?",
                exc,
            )
            raise
    else:
        raise ValueError(f"Unknown --rerun_mode: {rerun_mode} (use file, serve, or connect)")

    rr.log("/", rr.Clear(recursive=True))

    # Start from rest / initial state
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
    for frame_idx in tqdm(frame_range, desc="Replaying frames", unit="frame"):
        # Set controller targets for this frame pair
        simulator.set_controller_target(frame_idx, pure_inference=True)

        if simulator.object_collision_flag:
            simulator.update_collision_graph()

        # Advance one frame of physics (only this is timed; excludes setup + Rerun I/O)
        t_phys_start = time.perf_counter()
        if cfg.use_graph and hasattr(simulator, "forward_graph"):
            wp.capture_launch(simulator.forward_graph)
        else:
            simulator.step()
        t_physics_total += time.perf_counter() - t_phys_start

        # Log current state to Rerun (not included in physics timing)
        current_ctrl = controller_points[frame_idx]
        log_spring_mass_frame(
            simulator=simulator,
            frame_idx=frame_idx,
            controller_positions=current_ctrl,
            timeline="frame",
            options=opts,
        )

        # Use the last state as the starting point for the next frame
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
            pure_inference=True,
        )

    fps_physics = num_frames / t_physics_total if t_physics_total > 0 else 0.0
    ms_per_frame_physics = 1000.0 * t_physics_total / num_frames if num_frames > 0 else 0.0

    total_substeps = num_frames * num_substeps
    substeps_per_sec = total_substeps / t_physics_total if t_physics_total > 0 else 0.0

    print("\n" + "=" * 60, flush=True)
    print("  PHYS-TWIN REPLAY PERFORMANCE (physics forward only, excl. Rerun I/O)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Spring-mass model:", flush=True)
    print(f"    vertices: {n_vertices:,}   springs: {n_springs:,}   substeps/frame: {num_substeps}", flush=True)
    print(f"  Physics forward:", flush=True)
    print(f"    frames:       {num_frames} (frame {start_frame} .. {end_frame})", flush=True)
    print(f"    physics time: {t_physics_total:.2f} s", flush=True)
    print(f"    frame FPS:    {fps_physics:.1f} frames/s", flush=True)
    print(f"    per frame:    {ms_per_frame_physics:.2f} ms", flush=True)
    print(f"    substep rate: {substeps_per_sec:,.0f} substeps/s", flush=True)
    print("=" * 60 + "\n", flush=True)

    logger.info("[RERUN-REPLAY] Finished streaming to Rerun.")


def main() -> None:
    parser = ArgumentParser(description="Replay PhysTwin spring-mass simulations into Rerun.")
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
        help="Base path containing <case_name>/final_data.pkl etc.",
    )
    parser.add_argument(
        "--case_name",
        type=str,
        required=True,
        help="Name of the PhysTwin case to replay (e.g., double_lift_cloth_3).",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=1,
        help="First frame index to replay (default: 1).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to replay; omit for all frames (default: all).",
    )
    parser.add_argument(
        "--rerun_mode",
        type=str,
        choices=("file", "serve", "connect"),
        default="file",
        help="file (default): write .rrd; copy to laptop and rerun <path>. serve: serve_grpc. connect: connect_grpc.",
    )
    parser.add_argument(
        "--output-rrd",
        type=str,
        default=None,
        help="Output .rrd path for --rerun_mode file (default: replay_<case_name>.rrd).",
    )
    parser.add_argument(
        "--grpc_port",
        type=int,
        default=9876,
        help="Preferred port for rr.serve_grpc() when --rerun_mode=serve (default: 9876).",
    )
    parser.add_argument(
        "--auto-grpc-port",
        action="store_true",
        help="If --grpc_port is busy, try the next free port (default: off).",
    )
    parser.add_argument(
        "--strict-grpc-port",
        action="store_true",
        help="Exit if --grpc_port is busy (bind check). Default is to use --grpc_port as-is.",
    )
    parser.add_argument(
        "--connect_url",
        type=str,
        default=None,
        help="Optional URL for rr.connect_grpc() when --rerun_mode=connect "
        "(default: rerun+http://127.0.0.1:9876/proxy).",
    )
    parser.add_argument(
        "--no-velocities",
        action="store_true",
        help="Disable physics/velocities view.",
    )
    parser.add_argument(
        "--no-forces",
        action="store_true",
        help="Disable physics/forces view.",
    )
    parser.add_argument(
        "--no-spring-stretch",
        action="store_true",
        help="Disable physics/springs_stretch view.",
    )
    parser.add_argument(
        "--no-masses",
        action="store_true",
        help="Disable physics/masses view.",
    )
    parser.add_argument(
        "--no-collisions",
        action="store_true",
        help="Disable physics/collisions view.",
    )
    parser.add_argument(
        "--no-control-interpolation",
        action="store_true",
        help="Disable controls/interpolation view.",
    )
    parser.add_argument(
        "--no-ground-plane",
        action="store_true",
        help="Disable world/ground plane.",
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=0.05,
        help="Scale for velocity arrows (default: 0.05).",
    )
    parser.add_argument(
        "--force-scale",
        type=float,
        default=1e-4,
        help="Scale for force arrows (default: 1e-4).",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Log only original views (nodes, controls, springs). Disable physics/controls/ground views.",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Intermediate: original + spring stretch + masses + ground. Skip forces, velocities, collisions, control interpolation.",
    )
    parser.add_argument(
        "--no-global-normalization",
        action="store_true",
        help="Disable across-time color normalization (default: on). Use per-frame percentile instead.",
    )
    args = parser.parse_args()

    if args.minimal:
        logging_options = SpringMassLoggingOptions(
            velocities=False,
            forces=False,
            spring_stretch=False,
            masses=False,
            collisions=False,
            control_interpolation=False,
            ground_plane=False,
        )
    elif args.medium:
        logging_options = SpringMassLoggingOptions(
            velocities=False,
            forces=False,
            spring_stretch=True,
            masses=True,
            collisions=False,
            control_interpolation=False,
            ground_plane=True,
        )
    else:
        logging_options = SpringMassLoggingOptions(
            velocities=not args.no_velocities,
            forces=not args.no_forces,
            spring_stretch=not args.no_spring_stretch,
            masses=not args.no_masses,
            collisions=not args.no_collisions,
            control_interpolation=not args.no_control_interpolation,
            ground_plane=not args.no_ground_plane,
            velocity_scale=args.velocity_scale,
            force_scale=args.force_scale,
        )

    set_all_seeds(42)
    load_config_and_camera(args.base_path, args.case_name)
    trainer = load_trainer_and_model(args.base_path, args.case_name)
    replay_with_rerun(
        trainer=trainer,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        rerun_mode=args.rerun_mode,
        grpc_port=args.grpc_port,
        auto_grpc_port=args.auto_grpc_port,
        strict_grpc_port=args.strict_grpc_port,
        connect_url=args.connect_url,
        output_rrd=args.output_rrd,
        case_name=args.case_name,
        logging_options=logging_options,
        global_normalization=not args.no_global_normalization,
    )


if __name__ == "__main__":
    main()

