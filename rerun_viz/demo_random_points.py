#!/usr/bin/env python3
"""
Tiny Rerun demo: random 3D points, a wiggly line strip, and a few static axes markers.

No PhysTwin / Warp — only numpy + rerun-sdk.

**Default: ``--rerun-mode serve``** — VM runs ``serve_grpc``; on the laptop use SSH ``-L`` and
``rerun rerun+http://127.0.0.1:9876/proxy`` (recommended for VM → laptop live view).

**``--rerun-mode connect``** — JTAP-style ``connect_grpc`` + SSH ``-R`` (often flaky over SSH).

Example (VM, default serve):
  python -m rerun_viz.demo_random_points --seconds 30
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rerun as rr

from rerun_viz.connect_instructions import JTAP_STYLE_SSH, REMOTE_LIVE_SERVE_SSH
from rerun_viz.port_util import pick_free_port, port_free


def main() -> None:
    p = argparse.ArgumentParser(description="Random points Rerun demo (connect or serve).")
    p.add_argument(
        "--rerun-mode",
        choices=("connect", "serve"),
        default="serve",
        help="serve (default)=serve_grpc on VM; use SSH -L + rerun on laptop. connect=JTAP connect_grpc + -R.",
    )
    p.add_argument(
        "--connect-url",
        type=str,
        default=None,
        help="URL for rr.connect_grpc (default: SDK default rerun+http://127.0.0.1:9876/proxy).",
    )
    p.add_argument("--port", type=int, default=9876, help="Port for serve_grpc (serve mode only).")
    p.add_argument(
        "--auto-port",
        action="store_true",
        help="If set, scan for the next free port starting at --port (default: off; use 9876 as-is).",
    )
    p.add_argument(
        "--strict-port",
        action="store_true",
        help="Exit if --port is busy (after bind check). Implies no --auto-port.",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="How long to stream (default: 60). Use Ctrl+C to stop early.",
    )
    p.add_argument("--fps", type=float, default=20.0, help="Frames per second (default: 20)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    rr.init("rerun_demo_random_points", spawn=False)

    if args.rerun_mode == "connect":
        print(JTAP_STYLE_SSH, flush=True)
        rr.connect_grpc(url=args.connect_url)
        print("[rerun] Connected (connect mode). Logging random points...\n", flush=True)
    else:
        if args.strict_port and args.auto_port:
            print("ERROR: use only one of --strict-port or --auto-port", file=sys.stderr)
            sys.exit(1)

        if args.auto_port:
            port, used_fallback = pick_free_port(args.port)
            if used_fallback:
                print(
                    f"[rerun] Port {args.port} was busy; using {port} instead.\n"
                    f"        Use local forward to this host:{port} and:\n"
                    f"        rerun rerun+http://127.0.0.1:{port}/proxy\n",
                    file=sys.stderr,
                )
        elif args.strict_port:
            if not port_free(args.port):
                print(
                    f"ERROR: port {args.port} is in use.\n"
                    f"  lsof -i :{args.port}\n"
                    "  Or run with --auto-port to pick the next free port.",
                    file=sys.stderr,
                )
                sys.exit(1)
            port = args.port
        else:
            port = args.port

        print(REMOTE_LIVE_SERVE_SSH, flush=True)
        uri = rr.serve_grpc(grpc_port=port)
        print(f"\nServing: {uri}\n", flush=True)
        print(
            "On your LAPTOP (with ssh -L in place), run:\n"
            f"  rerun rerun+http://127.0.0.1:{port}/proxy\n",
            flush=True,
        )

    rr.log("/", rr.Clear(recursive=True))

    # Static origin + axis-colored reference points
    rr.log("demo/origin", rr.Points3D([[0, 0, 0]], radii=0.08, colors=[[255, 255, 255, 255]]))
    rr.log(
        "demo/axis_markers",
        rr.Points3D(
            positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            radii=0.04,
            colors=[[255, 80, 80, 255], [80, 255, 80, 255], [80, 80, 255, 255]],
        ),
    )

    dt = 1.0 / max(args.fps, 1e-6)
    n_frames = int(args.seconds * args.fps)
    t0 = time.time()
    path: list[np.ndarray] = []

    # Random walk for a line strip
    pos = np.zeros(3, dtype=np.float32)
    for frame in range(n_frames):
        rr.set_time("frame", sequence=frame)

        # Cloud of random points in a cube [-1,1]^3
        n_pts = 80
        pts = rng.uniform(-1.0, 1.0, size=(n_pts, 3)).astype(np.float32)
        # Per-point color from position (cheap RGBA)
        c = ((pts + 1.0) * 0.5 * 255.0).astype(np.uint8)
        colors = np.concatenate([c, np.full((n_pts, 1), 255, dtype=np.uint8)], axis=1)
        rr.log(
            "demo/random_cloud",
            rr.Points3D(positions=pts, radii=0.02, colors=colors),
        )

        # Wiggly trail
        pos = pos + rng.normal(0.0, 0.05, size=3).astype(np.float32)
        pos = np.clip(pos, -1.2, 1.2)
        path.append(pos.copy())
        trail = np.stack(path[-200:], axis=0)  # last 200 points
        rr.log(
            "demo/trail",
            rr.LineStrips3D(
                strips=[trail],
                colors=[255, 200, 50, 255],
                radii=0.008,
            ),
        )

        # Scalar / text for debugging
        rr.log("demo/stats/n_points", rr.Scalars(float(n_pts)))
        rr.log("demo/stats/frame", rr.Scalars(float(frame)))

        elapsed = time.time() - t0
        if elapsed >= args.seconds:
            break
        time.sleep(dt)

    print("Done. (Server stops when this process exits.)")


if __name__ == "__main__":
    main()
