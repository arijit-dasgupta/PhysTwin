"""Disk cache for replay global color ranges (speeds up pre-pass on repeat runs)."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from rerun_viz.spring_mass_logging import GlobalColorRanges

_CACHE_VERSION = 1


def prepass_cache_file_path(case_name: str, start_frame: int, end_frame: int) -> Path:
    return (
        Path("experiments")
        / case_name
        / "train"
        / f".replay_prepass_{start_frame}_{end_frame}.pkl"
    )


def _ranges_to_payload(gr: GlobalColorRanges) -> dict:
    return {
        "stiffness": list(gr.stiffness) if gr.stiffness is not None else None,
        "stretch": list(gr.stretch) if gr.stretch is not None else None,
        "mass": list(gr.mass) if gr.mass is not None else None,
    }


def _payload_to_ranges(d: dict) -> GlobalColorRanges:
    def _t(x: list | None) -> tuple[float, float] | None:
        if x is None:
            return None
        return (float(x[0]), float(x[1]))

    return GlobalColorRanges(
        stiffness=_t(d.get("stiffness")),
        stretch=_t(d.get("stretch")),
        mass=_t(d.get("mass")),
    )


def try_load_prepass_cache(
    case_name: str,
    start_frame: int,
    end_frame: int,
    checkpoint_path: str,
) -> GlobalColorRanges | None:
    path = prepass_cache_file_path(case_name, start_frame, end_frame)
    if not path.is_file():
        return None
    try:
        ck_abs = os.path.abspath(checkpoint_path)
        mtime = os.path.getmtime(ck_abs)
        with open(path, "rb") as f:
            p = pickle.load(f)
    except OSError:
        return None
    if p.get("v") != _CACHE_VERSION:
        return None
    if p.get("checkpoint_path") != os.path.abspath(checkpoint_path):
        return None
    if float(p.get("checkpoint_mtime", -1.0)) != float(mtime):
        return None
    if p.get("start_frame") != start_frame or p.get("end_frame") != end_frame:
        return None
    try:
        return _payload_to_ranges(p["ranges"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def save_prepass_cache(
    case_name: str,
    start_frame: int,
    end_frame: int,
    checkpoint_path: str,
    gr: GlobalColorRanges,
) -> Path | None:
    path = prepass_cache_file_path(case_name, start_frame, end_frame)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ck_abs = os.path.abspath(checkpoint_path)
        payload = {
            "v": _CACHE_VERSION,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "checkpoint_path": ck_abs,
            "checkpoint_mtime": os.path.getmtime(ck_abs),
            "ranges": _ranges_to_payload(gr),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path
    except OSError:
        return None
