#!/usr/bin/env python3
"""
Minimal Rerun connectivity test — no PhysTwin, no Warp.

Use this to verify that:
  1) rr.serve_grpc() can bind on the VM
  2) Your port-forward + viewer on the laptop can see the stream

If you see "Address already in use", pick another --port (e.g. 9877).

On the laptop (after SSH port-forward matches that port), run:
  rerun rerun+http://127.0.0.1:<port>/proxy

Then press Enter in this terminal to stop the server.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rerun as rr

from rerun_viz.port_util import pick_free_port, port_free


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal Rerun serve_grpc smoke test.")
    p.add_argument(
        "--port",
        type=int,
        default=9876,
        help="Preferred gRPC port (default: 9876).",
    )
    p.add_argument(
        "--auto-port",
        action="store_true",
        help="Scan for next free port from --port if busy (default: off).",
    )
    p.add_argument(
        "--strict-port",
        action="store_true",
        help="Exit if --port is busy (bind check).",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=120,
        help="How many fake frames to log (default: 120).",
    )
    args = p.parse_args()

    if args.strict_port and args.auto_port:
        print("ERROR: use only one of --strict-port or --auto-port", file=sys.stderr)
        sys.exit(1)

    if args.auto_port:
        port, used_fallback = pick_free_port(args.port)
        if used_fallback:
            print(
                f"[rerun] Port {args.port} busy; using {port}. "
                f"Match SSH forward + rerun URL to port {port}.\n",
                file=sys.stderr,
            )
    elif args.strict_port:
        if not port_free(args.port):
            print(
                f"ERROR: port {args.port} is already in use.\n"
                f"  lsof -i :{args.port}\n"
                "  Or use --auto-port to pick the next free port.",
                file=sys.stderr,
            )
            sys.exit(1)
        port = args.port
    else:
        port = args.port

    rr.init("minimal_rerun_serve_test", spawn=False)
    uri = rr.serve_grpc(grpc_port=port)
    print(f"\nServing at: {uri}\n")
    print(
        "On your laptop (with port-forward from this VM to localhost), run:\n"
        f"  rerun rerun+http://127.0.0.1:{port}/proxy\n"
        "\n"
        "If you use a different local forwarded port, substitute it in the URL.\n"
        "This script will log a few moving points for ~a few seconds.\n"
    )

    for t in range(args.frames):
        rr.set_time("frame", sequence=t)
        # Two points moving in a small circle — impossible to miss in 3D view
        angle = t * 0.1
        pos = np.array(
            [
                [np.cos(angle), np.sin(angle), 0.0],
                [-np.cos(angle), -np.sin(angle), 0.5],
            ],
            dtype=np.float32,
        )
        rr.log("test/points", rr.Points3D(positions=pos, radii=0.05))
        time.sleep(1.0 / 30.0)

    print("Done logging. You can close the viewer or Ctrl+C this process.")
    # Keep process alive briefly so late-connecting viewers still get buffered data
    try:
        time.sleep(2.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
