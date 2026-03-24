"""CLI: compare full-res + downsampled spring-mass models in one Rerun recording."""

from __future__ import annotations

from argparse import ArgumentParser

import warp as wp  # noqa: F401 — before qqtt, quiet init

wp.config.quiet = True

from rerun_viz.multi_model_replay import replay_multiple_models_with_rerun  # noqa: E402


def main() -> None:
    parser = ArgumentParser(
        description="Replay full-res + all valid downsampled bundles: shared controls + meshes only."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
        help="Base path containing <case_name>/final_data.pkl etc.",
    )
    parser.add_argument("--case_name", type=str, required=True, help="PhysTwin case name.")
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Downsample tags to include (dirs under downsampled/<tag>/). "
        "Omit to auto-scan all valid bundles. Pass zero values ( --tags ) for full-res only.",
    )
    parser.add_argument("--start_frame", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--rerun_mode",
        type=str,
        choices=("file", "serve", "connect"),
        default="file",
    )
    parser.add_argument("--output-rrd", type=str, default=None)
    parser.add_argument("--grpc_port", type=int, default=9876)
    parser.add_argument("--auto-grpc-port", action="store_true")
    parser.add_argument("--strict-grpc-port", action="store_true")
    parser.add_argument("--connect_url", type=str, default=None)
    parser.add_argument(
        "--mesh-use-original-only",
        action="store_true",
        help="Mesh only tracked object vertices (not full surface shell).",
    )
    parser.add_argument(
        "--mesh-spring-max-edge-factor",
        type=float,
        default=6.0,
        help="Filter spring 3-clique triangles whose longest edge exceeds factor × median OO edge.",
    )
    parser.add_argument(
        "--mesh-max-triangles",
        type=int,
        default=0,
        help="Cap triangles per mesh after spring build (largest faces kept at rest pose; "
        "preserves sim vertex indices). Default 0 = no cap (previous behavior).",
    )
    parser.add_argument(
        "--mesh-opacity",
        type=float,
        default=0.25,
        help="Mesh opacity in [0, 1], applied via Rerun Mesh3D albedo_factor (default 0.25).",
    )
    parser.add_argument(
        "--mesh-alpha",
        type=int,
        default=None,
        help="If set, overrides --mesh-opacity using alpha 0–255 (mapped to [0, 1]).",
    )
    args = parser.parse_args()

    tags_arg = args.tags
    if tags_arg is not None and len(tags_arg) == 0:
        tags_arg = ()

    replay_multiple_models_with_rerun(
        args.base_path,
        args.case_name,
        tags=tags_arg,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        rerun_mode=args.rerun_mode,
        grpc_port=args.grpc_port,
        auto_grpc_port=args.auto_grpc_port,
        strict_grpc_port=args.strict_grpc_port,
        connect_url=args.connect_url,
        output_rrd=args.output_rrd,
        mesh_use_original_only=args.mesh_use_original_only,
        mesh_spring_max_edge_factor=args.mesh_spring_max_edge_factor,
        mesh_max_triangles=args.mesh_max_triangles,
        mesh_opacity=args.mesh_opacity,
        mesh_alpha=args.mesh_alpha,
    )


if __name__ == "__main__":
    main()
