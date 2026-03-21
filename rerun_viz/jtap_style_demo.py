#!/usr/bin/env python3
"""
Minimal Rerun smoke test: bright colored points + static axis markers.

**``Flush timed out``** means the viewer did **not** finish reading the gRPC stream.
The tunnel can show "TCP OK" while the laptop process on port 9876 is **not** a compatible
Rerun viewer (wrong app, or viewer/SDK version mismatch). Fix: install the
**same viewer major version** as ``python -c "import rerun as rr; print(rr.version())"``,
or use **serve** mode, or use **--write-rrd** and open the file locally.

Modes:

- ``serve`` (default): ``serve_grpc`` on VM; laptop uses ``ssh -L`` + ``rerun rerun+http://...``.
- ``connect``: ``GrpcSink`` + SSH ``-R`` (JTAP-style; often flaky over SSH).
- ``file``: no network — writes ``.rrd``; copy to laptop and ``rerun file.rrd``.

Examples::

  # Prove logging works without network (VM):
  python -m rerun_viz.jtap_style_demo --mode file --rrd-path /tmp/jtap.rrd
  # scp to laptop, then:  rerun /tmp/jtap.rrd

  # Tee .rrd while using connect mode (viewer broken? open the .rrd on laptop):
  python -m rerun_viz.jtap_style_demo --mode connect --write-rrd /tmp/jtap.rrd
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import time

import numpy as np
import rerun as rr

from rerun_viz.connect_instructions import JTAP_STYLE_SSH, REMOTE_LIVE_SERVE_SSH

_DEFAULT_GRPC_PORT = 9876


def _tcp_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _warn_if_proxy_unreachable(port: int) -> None:
    if not _tcp_open("127.0.0.1", port):
        print(
            f"\n[rerun] WARNING: nothing accepts TCP on 127.0.0.1:{port} on this machine.\n"
            f"  Remote-forward (-R) is probably missing or failed.\n"
            f"  Or use: --mode serve  OR  --mode file\n",
            flush=True,
        )
    else:
        print(
            f"[rerun] OK: TCP to 127.0.0.1:{port} succeeds (tunnel port is open).\n"
            f"      If you still see no data / flush timeout, see top of this script.\n",
            flush=True,
        )


def _assert_rerun_enabled() -> None:
    if rr.is_enabled():
        return
    rerun_env = os.environ.get("RERUN", "(unset)")
    print(
        "\n[rerun] ERROR: Rerun recording is DISABLED.\n"
        f"  Environment RERUN={rerun_env!r}\n"
        "  Fix:  export RERUN=on\n",
        flush=True,
    )
    sys.exit(1)


def _flush_recording(*, timeout_sec: float) -> None:
    rs = rr.get_global_data_recording()
    if rs is None:
        return
    try:
        rs.flush(timeout_sec=timeout_sec)
    except Exception as e:
        print(
            f"\n[rerun] flush warning: {e}\n"
            "\n"
            "This usually means the **viewer is not consuming the gRPC stream** (not a network\n"
            "'TCP OK' check). Common causes:\n"
            "  • Viewer on laptop is **not** the Rerun app, or **wrong version** vs this SDK.\n"
            "  • Compare:  rerun --version   with:\n"
            f"        python -c \"import rerun as rr; print(rr.version())\"\n"
            "    → install matching viewer (https://github.com/rerun-io/rerun/releases).\n"
            "  • Try **serve** mode on this VM + ``ssh -L`` + ``rerun rerun+http://127.0.0.1:9876/proxy`` on laptop.\n"
            "  • Or run with **--write-rrd /tmp/x.rrd** and open the file on the laptop.\n",
            flush=True,
        )


def _setup_sinks_connect(url: str, write_rrd: str | None) -> None:
    """Prefer set_sinks so we can tee to a file when gRPC is flaky."""
    if write_rrd:
        rr.set_sinks(rr.GrpcSink(url=url), rr.FileSink(write_rrd))
        print(f"[rerun] Also writing recording to: {write_rrd}\n", flush=True)
    else:
        rr.connect_grpc(url=url)


def _log_demo_frames(rng: np.random.Generator, *, n_frames: int, interrupted: list[bool]) -> None:
    rr.log("/", rr.Clear(recursive=True))

    rr.log("test/origin", rr.Points3D([[0, 0, 0]], radii=0.12, colors=[[255, 255, 255, 255]]))
    rr.log(
        "test/axis_markers",
        rr.Points3D(
            positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            radii=0.06,
            colors=[[255, 80, 80, 255], [80, 255, 80, 255], [80, 80, 255, 255]],
        ),
    )

    for t in range(n_frames):
        if interrupted[0]:
            break
        rr.set_time("frame", sequence=t)

        pts = rng.uniform(-1.0, 1.0, size=(40, 3)).astype(np.float32)
        c = ((pts + 1.0) * 0.5 * 255.0).astype(np.uint8)
        colors = np.concatenate([c, np.full((40, 1), 255, dtype=np.uint8)], axis=1)
        rr.log(
            "test/random_cloud",
            rr.Points3D(positions=pts, radii=0.04, colors=colors),
        )

        rr.log(
            "test/status",
            rr.TextLog(f"jtap_style_demo frame {t} — if you see this text, logging works."),
        )
        rr.log("test/scalars/frame", rr.Scalars(float(t)))

        time.sleep(1.0 / 20.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser(description="Minimal Rerun connect / serve / file smoke test.")
    p.add_argument(
        "--mode",
        choices=("connect", "serve", "file"),
        default="serve",
        help="serve (default)=serve_grpc on VM — use SSH -L + rerun on laptop. connect=GrpcSink + -R. file=.rrd only.",
    )
    p.add_argument("--grpc-port", type=int, default=_DEFAULT_GRPC_PORT, help="Port for connect/serve (default 9876).")
    p.add_argument("--frames", type=int, default=60, help="Number of frames to log.")
    p.add_argument(
        "--write-rrd",
        type=str,
        default=None,
        metavar="PATH",
        help="Also tee logs to this .rrd file (connect/serve only). Open on laptop if gRPC fails.",
    )
    p.add_argument(
        "--rrd-path",
        type=str,
        default="/tmp/jtap_style_demo.rrd",
        help="Output path for --mode file (default: /tmp/jtap_style_demo.rrd).",
    )
    p.add_argument(
        "--no-hold",
        action="store_true",
        help="In serve mode only: exit immediately after logging (default: keep server alive until Ctrl+C).",
    )
    args = p.parse_args()

    interrupted = [False]

    def _on_sigint(_signum, _frame) -> None:
        interrupted[0] = True
        print("\n[rerun] Interrupted — shutting down…", flush=True)

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"[rerun] SDK: {rr.version()}\n", flush=True)

    if args.mode == "connect":
        print(JTAP_STYLE_SSH, flush=True)
    elif args.mode == "serve":
        print(REMOTE_LIVE_SERVE_SSH, flush=True)

    rr.init("jtap_style_demo", spawn=False, default_enabled=True)
    _assert_rerun_enabled()

    if args.mode == "file":
        rr.save(args.rrd_path)
        print(f"[rerun] Writing only to file (no viewer): {args.rrd_path}\n", flush=True)
    elif args.mode == "connect":
        url = f"rerun+http://127.0.0.1:{args.grpc_port}/proxy"
        _setup_sinks_connect(url, args.write_rrd)
        _warn_if_proxy_unreachable(args.grpc_port)
    else:
        if args.write_rrd:
            print(
                "[rerun] Note: --write-rrd is only supported with --mode connect (tee via set_sinks).\n"
                "      For a file-only test use:  --mode file --rrd-path …\n",
                flush=True,
            )
        uri = rr.serve_grpc(grpc_port=args.grpc_port)
        print(
            f"\n[rerun] serve_grpc listening. On LAPTOP (ssh -L {args.grpc_port}:127.0.0.1:{args.grpc_port} …), run:\n"
            f"  rerun {uri}\n"
            "\n"
            ">>> Connect the viewer NOW (or any time while this process runs).\n"
            ">>> When this script exits, the server stops — you will see nothing if you connect too late.\n",
            flush=True,
        )

    rng = np.random.default_rng(0)

    try:
        _log_demo_frames(rng, n_frames=args.frames, interrupted=interrupted)

        # serve_grpc dies when this process exits — keep alive until Ctrl+C by default.
        if args.mode == "serve" and not args.no_hold and not interrupted[0]:
            print(
                "\n[rerun] Streaming finished. Server alive — connect viewer, scrub data, or Ctrl+C to exit.\n",
                flush=True,
            )
            while not interrupted[0]:
                time.sleep(0.25)
    finally:
        # Match minimal_rerun_serve_test: no disconnect/shutdown — let process exit cleanly.
        # Explicit disconnect/shutdown was breaking the stream for some setups.
        pass

    if args.mode == "file":
        print(
            f"\nDone. Copy {args.rrd_path} to your laptop and run:\n"
            f"  rerun {args.rrd_path}\n",
            flush=True,
        )
    else:
        print(
            "\nDone logging.\n"
            "  In Rerun: pick application **jtap_style_demo**; check **test/random_cloud** and **test/status**.\n",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
