"""Minimal, colorized CLI output for Rerun replay (TTY-aware; plain when piped)."""

from __future__ import annotations

import sys
from typing import TextIO


def _use_color(stream: TextIO) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[35m"


def _wrap(s: str, *codes: str, stream: TextIO | None = None) -> str:
    st = stream or sys.stdout
    if not _use_color(st):
        return s
    return "".join(codes) + s + _C.RESET


def print_replay_banner(case_name: str | None, mode: str, *, stream: TextIO | None = None) -> None:
    st = stream or sys.stdout
    title = _wrap(" PhysTwin · Rerun ", _C.BOLD, _C.CYAN, stream=st)
    sub = _wrap(f" {mode} ", _C.DIM, stream=st)
    case = f" · {_wrap(case_name or '?', _C.GREEN, stream=st)}" if case_name else ""
    print(f"\n{title}{sub}{case}\n", file=st, flush=True)


def print_file_recording(path: str, *, stream: TextIO | None = None) -> None:
    st = stream or sys.stdout
    p = _wrap(path, _C.BOLD, _C.GREEN, stream=st)
    hint = _wrap("rerun", _C.YELLOW, stream=st)
    print(
        f"  {_wrap('out', _C.DIM, stream=st)}  {p}",
        file=st,
        flush=True,
    )
    print(
        f"  {_wrap('→', _C.DIM, stream=st)}  {hint} {path}",
        file=st,
        flush=True,
    )


def print_prepass_cache_hit(cache_path: str, *, stream: TextIO | None = None) -> None:
    st = stream or sys.stdout
    p = _wrap(cache_path, _C.GREEN, stream=st)
    line = _wrap("pre-pass: using cached color ranges (delete file to recompute)", _C.DIM, stream=st)
    print(f"  {line}\n  {_wrap('cache', _C.DIM, stream=st)}  {p}", file=st, flush=True)


def print_dim_status(msg: str, *, stream: TextIO | None = None) -> None:
    st = stream or sys.stdout
    line = _wrap(f"·  {msg}", _C.DIM, stream=st)
    print(f"  {line}", file=st, flush=True)


def print_warning_line(msg: str, *, stream: TextIO | None = None) -> None:
    st = stream or sys.stdout
    line = _wrap(msg, _C.YELLOW, stream=st)
    print(f"  {line}", file=st, flush=True)


def print_ssh_hint_block(text: str, *, stream: TextIO | None = None) -> None:
    st = stream or sys.stdout
    dim = _wrap(text.strip(), _C.DIM, stream=st)
    print(dim, file=st, flush=True)


def print_serve_ready(port: int, uri: str, *, stream: TextIO | None = None) -> None:
    st = stream or sys.stdout
    port_s = _wrap(str(port), _C.GREEN, stream=st)
    url = _wrap(f"rerun rerun+http://127.0.0.1:{port}/proxy", _C.BOLD, _C.YELLOW, stream=st)
    print(
        f"  {_wrap('gRPC', _C.DIM, stream=st)}  {port_s}  {_wrap(uri, _C.DIM, stream=st)}",
        file=st,
        flush=True,
    )
    print(f"  {_wrap('→', _C.DIM, stream=st)}  {url}", file=st, flush=True)


def print_phys_twin_ready_panel(
    *,
    case_name: str,
    checkpoint_path: str,
    n_vertices: int,
    n_springs: int,
    num_substeps: int,
    num_frames: int,
    object_collision: bool,
    use_graph: bool,
    data_type: str,
    dt: float,
    stream: TextIO | None = None,
) -> None:
    """Pretty-print trained PhysTwin / simulator stats after load (before torch.load warning noise)."""
    st = stream or sys.stdout
    title = _wrap(" PhysTwin ready ", _C.BOLD, _C.CYAN, stream=st)
    case = _wrap(case_name, _C.GREEN, stream=st)
    ck = _wrap(checkpoint_path, _C.DIM, stream=st)
    y_n = _wrap("yes" if object_collision else "no", _C.YELLOW if object_collision else _C.DIM, stream=st)
    g = _wrap("yes" if use_graph else "no", _C.GREEN if use_graph else _C.DIM, stream=st)
    dt_s = _wrap(f"{dt:.6f}", _C.MAGENTA, stream=st)

    print(f"\n{title}  {case}\n", file=st, flush=True)
    print(f"  {_wrap('checkpoint', _C.DIM, stream=st)}  {ck}", file=st, flush=True)
    print(
        f"  {_wrap('data', _C.DIM, stream=st)}       {_wrap(data_type, _C.DIM, stream=st)}  ·  "
        f"{_wrap('Δt', _C.DIM, stream=st)} {dt_s} s  ·  "
        f"{_wrap('CUDA graph', _C.DIM, stream=st)} {g}",
        file=st,
        flush=True,
    )
    print(
        f"  {_wrap('mesh', _C.DIM, stream=st)}       "
        f"{n_vertices:,} verts  ·  {n_springs:,} springs  ·  {num_substeps} substeps/step",
        file=st,
        flush=True,
    )
    print(
        f"  {_wrap('sequence', _C.DIM, stream=st)}   {num_frames:,} controller frames  ·  "
        f"{_wrap('object-object collision', _C.DIM, stream=st)} {y_n}",
        file=st,
        flush=True,
    )
    print(file=st, flush=True)


def print_performance_summary(
    *,
    n_vertices: int,
    n_springs: int,
    num_substeps: int,
    num_frames: int,
    start_frame: int,
    end_frame: int,
    t_physics_total: float,
    stream: TextIO | None = None,
) -> None:
    st = stream or sys.stdout
    fps = num_frames / t_physics_total if t_physics_total > 0 else 0.0
    ms_pf = 1000.0 * t_physics_total / num_frames if num_frames > 0 else 0.0
    total_substeps = num_frames * num_substeps
    sps = total_substeps / t_physics_total if t_physics_total > 0 else 0.0

    head = _wrap(" physics ", _C.BOLD, _C.CYAN, stream=st)
    print(f"\n{head}", file=st, flush=True)
    print(
        f"  {_wrap('frames', _C.DIM, stream=st)}  {start_frame}…{end_frame}  ({num_frames})",
        file=st,
        flush=True,
    )
    print(
        f"  {_wrap('mesh', _C.DIM, stream=st)}   "
        f"{n_vertices:,} verts · {n_springs:,} springs · {num_substeps} substeps/frame",
        file=st,
        flush=True,
    )
    fps_s = _wrap(f"{fps:.1f} fps", _C.GREEN, stream=st)
    print(
        f"  {_wrap('time', _C.DIM, stream=st)}   "
        f"{t_physics_total:.2f}s  ·  {fps_s}  ·  {ms_pf:.1f} ms/frame  ·  {sps:,.0f} substeps/s",
        file=st,
        flush=True,
    )
    print(file=st, flush=True)
