"""CLI entry: replay PhysTwin spring-mass simulations into Rerun (.rrd, serve, or connect)."""

from __future__ import annotations

from argparse import ArgumentParser

# Import Warp before qqtt so the device banner / module-load timers respect quiet mode.
import warp as wp  # noqa: E402

wp.config.quiet = True

from qqtt.utils import cfg  # noqa: E402
from rerun_viz.replay_core import (  # noqa: E402
    load_config_and_camera,
    load_trainer_and_model,
    quiet_qqtt_stream_logs,
    replay_with_rerun,
    set_all_seeds,
)
from rerun_viz.spring_mass_logging import SpringMassLoggingOptions  # noqa: E402
from rerun_viz.terminal_output import print_phys_twin_ready_panel  # noqa: E402


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
    parser.add_argument(
        "--no-prepass-cache",
        action="store_true",
        help="Do not read/write experiments/<case>/train/.replay_prepass_<start>_<end>.pkl (always run full pre-pass).",
    )
    parser.add_argument(
        "--prepass-refresh",
        action="store_true",
        help="Ignore pre-pass cache and recompute (overwrites cache when enabled).",
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
    with quiet_qqtt_stream_logs():
        load_config_and_camera(args.base_path, args.case_name)
        trainer, checkpoint_path = load_trainer_and_model(
            args.base_path, args.case_name, return_checkpoint_path=True
        )

    sim = trainer.simulator
    controller_points = sim.controller_points
    assert controller_points is not None
    print_phys_twin_ready_panel(
        case_name=args.case_name,
        checkpoint_path=checkpoint_path,
        n_vertices=sim.n_vertices,
        n_springs=sim.n_springs,
        num_substeps=sim.num_substeps,
        num_frames=int(controller_points.shape[0]),
        object_collision=bool(sim.object_collision_flag),
        use_graph=bool(cfg.use_graph),
        data_type=str(cfg.data_type),
        dt=float(cfg.dt),
    )

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
        checkpoint_path=checkpoint_path,
        prepass_cache=not args.no_prepass_cache,
        prepass_refresh=args.prepass_refresh,
    )


if __name__ == "__main__":
    main()
