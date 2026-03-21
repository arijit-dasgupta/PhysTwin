from __future__ import annotations

import glob
import json
import os
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import rerun as rr
import torch
import warp as wp

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger
from rerun_viz.connect_instructions import JTAP_STYLE_SSH, REMOTE_LIVE_SERVE_SSH
from rerun_viz.port_util import pick_free_port, port_free
from rerun_viz.spring_mass_logging import log_spring_mass_frame


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

    for frame_idx in range(start_frame, end_frame + 1):
        # Set controller targets for this frame pair
        simulator.set_controller_target(frame_idx, pure_inference=True)

        if simulator.object_collision_flag:
            simulator.update_collision_graph()

        # Advance one frame of physics
        if cfg.use_graph and hasattr(simulator, "forward_graph"):
            wp.capture_launch(simulator.forward_graph)
        else:
            simulator.step()

        # Log current state to Rerun
        current_ctrl = controller_points[frame_idx]
        log_spring_mass_frame(
            simulator=simulator,
            frame_idx=frame_idx,
            controller_positions=current_ctrl,
            timeline="frame",
        )

        # Use the last state as the starting point for the next frame
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
            pure_inference=True,
        )

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
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()

