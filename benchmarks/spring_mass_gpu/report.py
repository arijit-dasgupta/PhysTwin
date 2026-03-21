"""Write unified Markdown report + matplotlib PNGs."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.spring_mass_gpu.profile_segments import derived_outer_ms_per_substep

# Order for segment plots / tables (matches ``SpringMassSystemWarp.step()`` instrumentation).
SEGMENT_ORDER = (
    "control_points",
    "springs",
    "velocity_integration",
    "object_pair_collision",
    "integrate_ground",
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _md_fenced_code(text: str) -> str:
    """Wrap *text* in a markdown fenced block (escape close fence if present)."""
    safe = text.replace("```", "`\u200b``")
    return f"```\n{safe}\n```\n"


def _scatter_track_b(
    fig_dir: str,
    filename: str,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    xlabel: str,
    title: str,
    ylabel: str = "Outer step mean (ms), one full step()",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys)
    for x, y, lb in zip(xs, ys, labels, strict=True):
        ax.annotate(lb, (x, y), fontsize=7, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_segment_track_b(
    fig_dir: str,
    segment: str,
    topology_rows: list[Any],
    x_key: str,
    xlabel: str,
    filename: str,
) -> str | None:
    rows = [r for r in topology_rows if getattr(r, "segment_ms", None) and segment in r.segment_ms]
    if len(rows) < 1:
        return None
    xs = [float(getattr(r.size, x_key)) for r in rows]
    ys = [float(r.segment_ms[segment]) for r in rows]
    labels = [r.case_name for r in rows]
    title = f"Track (B): segment “{segment}” (eager, ms summed over substeps) vs {xlabel}"
    path = _scatter_track_b(
        fig_dir,
        filename,
        xs,
        ys,
        labels,
        xlabel,
        title,
        ylabel=f"Segment {segment} (ms), one outer step, eager",
    )
    return path


def _plot_segment_track_a(
    fig_dir: str,
    segment: str,
    parallel_rows: list[Any],
    filename: str,
) -> str | None:
    rows = [r for r in parallel_rows if getattr(r, "segment_ms", None) and segment in r.segment_ms]
    if len(rows) < 2:
        return None
    ns = [r.n_instances for r in rows]
    ys = [float(r.segment_ms[segment]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, ys, marker="o")
    ax.set_xlabel("N (sequential simulators in one timed round)")
    ax.set_ylabel(f"Sum of segment “{segment}” over N steps (ms)")
    ax.set_title(
        f"Track (A): segment “{segment}” vs N (eager; sum matches per-round segment total)"
    )
    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_report(
    output_dir: str,
    *,
    gpu: Any,
    topology_rows: list[Any],
    parallel_rows: list[Any] | None,
    trainers_count: int,
    oom_at_n: int | None,
    parallel_case: str,
    base_path: str,
    warmup: int,
    repeats: int,
    extra_notes: str = "",
    event_log: list[str] | None = None,
    error_summary: str | None = None,
    parallel_timing_complete: bool = True,
    parallel_load_complete: bool = True,
    timing_row_count: int | None = None,
    last_timed_n: int | None = None,
    segment_profile_enabled: bool = False,
    benchmark_use_graph: bool = True,
) -> str:
    import torch
    import warp as wp

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    fig_b = os.path.join(fig_dir, "track_b")
    fig_a = os.path.join(fig_dir, "track_a")
    os.makedirs(fig_b, exist_ok=True)
    os.makedirs(fig_a, exist_ok=True)

    lines: list[str] = []
    lines.append("# PhysTwin spring-mass GPU benchmark report\n")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}  \n")
    lines.append(f"**Git:** `{_git_sha()}`  \n\n")

    lines.append("## Environment\n\n")
    lines.append("| Field | Value |\n|---|---|\n")
    lines.append(f"| GPU index | {gpu.device_index} |\n")
    lines.append(f"| Device name | {gpu.device_name} |\n")
    lines.append(
        f"| `get_device_properties().total_memory` | {gpu.properties_total_memory_bytes} bytes |\n"
    )
    lines.append(f"| `mem_get_info` total | {gpu.mem_get_info_total_bytes} bytes |\n")
    lines.append(f"| `mem_get_info` free | {gpu.mem_get_info_free_bytes} bytes |\n")
    lines.append(f"| Sanity (totals match) | **{gpu.sanity_total_mem_match}** |\n")
    lines.append(f"| CUDA (torch.version.cuda) | {gpu.cuda_version} |\n")
    lines.append(f"| PyTorch | {torch.__version__} |\n")
    lines.append(f"| Warp | {getattr(wp, '__version__', 'unknown')} |\n")
    if gpu.nvidia_smi_line:
        lines.append(f"| nvidia-smi | `{gpu.nvidia_smi_line}` |\n")
    lines.append("\n")

    if event_log:
        lines.append("## Run log\n\n")
        for line in event_log:
            lines.append(f"- {line}\n")
        lines.append("\n")

    if error_summary:
        lines.append("## Run failure\n\n")
        lines.append("The benchmark exited with an error. Details:\n\n")
        lines.append(_md_fenced_code(error_summary))

    lines.append("## Methodology\n\n")
    lines.append(
        "- **Forward only:** one **outer** physics update per timed interval (see glossary). "
        "When `cfg.use_graph` is True, the outer step is a **CUDA graph replay** of the same work "
        "as `SpringMassSystemWarp.step()` (`wp.capture_launch(forward_graph)`). When `False`, the "
        "outer step is a direct `step()` call.\n"
    )
    lines.append(
        "- **Outer step vs substep:** A **substep** is one index `i` in the inner loop "
        "`for i in range(num_substeps)` inside `step()`. One **outer step** is the **entire** call "
        "to `step()` — including **all** substeps (and graph capture wraps that full inner loop). "
        "Table column **“Outer step mean (ms)”** is one mean wall time for that full outer step.\n"
    )
    lines.append(
        "- **Derived column:** **outer ms per substep** = `outer_step_mean_ms / num_substeps` "
        "(average duration per inner substep if work were uniform; no separate substep timer).\n"
    )
    lines.append(
        "- **Collision:** production-like — loaded from case + `cfg` (see per-row flags).\n"
    )
    lines.append(f"- **Data root:** `{base_path}`\n")
    lines.append(f"- **Warmup / repeats:** {warmup} / {repeats}\n")
    lines.append(
        f"- **Primary benchmark path `use_graph`:** **{benchmark_use_graph}** (same as `cfg.use_graph` during the run).\n"
    )
    if segment_profile_enabled:
        lines.append(
            "- **Segment timing (optional):** CUDA events inside `step()` with **`cfg.use_graph=False`** "
            "for that measurement. Segment sums are **eager-only** and are **not** equal to graph-mode "
            "wall time; use them to see where work goes inside the eager pipeline.\n"
        )
    oom_note = (
        str(oom_at_n)
        if oom_at_n is not None
        else "*(pending — load phase not finished or report written mid-load)*"
    )
    lines.append(
        f"- **Track (A) load ceiling:** `{parallel_case}` — load independent `InvPhyTrainerWarp` "
        f"instances until first load-time OOM (`oom_at_n = {oom_note}`). This is **not** the same "
        "as how many simulators can be **stepped** in one timing sweep.\n"
    )
    lines.append(
        "- **Track (A) timing sweep:** For `n = 1 .. K`, one timed round **sequentially** steps the "
        "first `n` trainers once each. Peak activation can OOM before `n` reaches the load ceiling — "
        "so **timing rows** can stop **before** `trainers_count`.\n"
    )
    lines.append("- **Track (B) topology:** one row per discovered case; different mesh sizes.\n\n")
    if extra_notes:
        lines.append(f"{extra_notes}\n\n")

    lines.append("## Glossary (locked terminology)\n\n")
    lines.append("| Term | Meaning |\n|---|---|\n")
    lines.append(
        "| **Outer step** | One call to `SpringMassSystemWarp.step()` **or** one captured graph launch "
        "that wraps the same work — **entire** inner substep loop. |\n"
    )
    lines.append(
        "| **Substep** | Single iteration `i` inside `step()`; there are `num_substeps` per outer step. |\n"
    )
    lines.append(
        "| **Track (A) — load ceiling** | Max trainer instances successfully **constructed** before "
        "first load-time OOM: `trainers_count`, `oom_at_n` (1-based index of failing load). |\n"
    )
    lines.append(
        "| **Track (A) — timing sweep** | For `n = 1 .. K`, time one sequential round stepping the "
        "first `n` trainers once. `K` ≤ `trainers_count`; if stepping OOMs early, `timing_complete` "
        "is false and there are fewer rows than loads. |\n"
    )
    lines.append(
        "| **Per-instance time (Track A)** | `total_round_ms / n` for that timed row (`per_instance_ms`). |\n"
    )
    lines.append(
        "| **Segment time (optional)** | Eager-only breakdown of one outer `step()` into pipeline "
        "stages; not equivalent to graph replay wall time. |\n"
    )
    lines.append("\n")

    lines.append("## Track (B) — topology / per-case\n\n")
    lines.append(
        "| case | n_vertices | num_object_points | n_springs | substeps | outer ms per substep (derived) | "
        "coll. | **Outer step mean (ms)** — one full `step()` | std (ms) | spring_Y MB (approx) | peak alloc MB |\n"
        "|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|\n"
    )
    for row in topology_rows:
        s = row.size
        coll = f"{s.object_collision_flag} / cfg.self_collision={s.self_collision_cfg}"
        sy_mb = s.spring_Y_bytes / (1024.0**2)
        der = derived_outer_ms_per_substep(row.timing.mean_ms, s.num_substeps)
        lines.append(
            f"| {s.case_name} | {s.n_vertices} | {s.num_object_points} | {s.n_springs} | {s.num_substeps} | "
            f"{der:.6f} | {coll} | {row.timing.mean_ms:.4f} | {row.timing.std_ms:.4f} | "
            f"{sy_mb:.4f} | {row.peak_mem_mb:.2f} |\n"
        )
    lines.append("\n")

    if len(topology_rows) >= 2:
        xs_sp = [float(r.size.n_springs) for r in topology_rows]
        xs_op = [float(r.size.num_object_points) for r in topology_rows]
        xs_nv = [float(r.size.n_vertices) for r in topology_rows]
        ys = [r.timing.mean_ms for r in topology_rows]
        labels = [r.case_name for r in topology_rows]
        _scatter_track_b(
            fig_b,
            "track_b_time_vs_springs.png",
            xs_sp,
            ys,
            labels,
            "n_springs",
            "Track (B): outer step mean vs spring count",
        )
        lines.append(
            "![Track B1 — springs](figures/track_b/track_b_time_vs_springs.png)\n\n"
        )
        lines.append(
            "**Figure B1.** Each point is one real case. **Y** is **outer step mean (ms)** — one full "
            "`step()` including all substeps (graph path when `use_graph` is enabled for the benchmark). "
            "**X** is `n_springs`.\n\n"
        )

        _scatter_track_b(
            fig_b,
            "track_b_time_vs_object_points.png",
            xs_op,
            ys,
            labels,
            "num_object_points",
            "Track (B): outer step mean vs object points",
        )
        lines.append(
            "![Track B2 — object points](figures/track_b/track_b_time_vs_object_points.png)\n\n"
        )
        lines.append(
            "**Figure B2.** Same outer-step mean as B1. **X** is `num_object_points` (mesh points on the object).\n\n"
        )

        _scatter_track_b(
            fig_b,
            "track_b_time_vs_vertices.png",
            xs_nv,
            ys,
            labels,
            "n_vertices",
            "Track (B): outer step mean vs vertex count",
        )
        lines.append(
            "![Track B3 — vertices](figures/track_b/track_b_time_vs_vertices.png)\n\n"
        )
        lines.append(
            "**Figure B3.** Same outer-step mean. **X** is `n_vertices` (full vertex count).\n\n"
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(xs_sp, ys, "o")
        for x, y, lb in zip(xs_sp, ys, labels, strict=True):
            ax.annotate(lb, (x, y), fontsize=7, alpha=0.85)
        ax.set_xlabel("n_springs")
        ax.set_ylabel("Outer step mean (ms)")
        ax.set_title("Track (B): log–log (optional) — springs vs time")
        fig.tight_layout()
        p_b4 = os.path.join(fig_b, "track_b_time_vs_springs_loglog.png")
        fig.savefig(p_b4, dpi=120)
        plt.close(fig)
        lines.append(
            "![Track B4 — loglog](figures/track_b/track_b_time_vs_springs_loglog.png)\n\n"
        )
        lines.append(
            "**Figure B4 (optional).** Log–log view of B1 for rough scaling intuition.\n\n"
        )

    if segment_profile_enabled and topology_rows:
        rep = next((r for r in topology_rows if getattr(r, "segment_ms", None)), None)
        if rep and rep.segment_ms:
            keys = [k for k in SEGMENT_ORDER if k in rep.segment_ms]
            vals = [float(rep.segment_ms[k]) for k in keys]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(keys, vals, color="tab:blue", alpha=0.85)
            ax.set_ylabel("ms (eager, summed over substeps)")
            ax.set_title(
                f"Segment breakdown (one case: {rep.case_name}) — eager-only; use_graph=False for measurement"
            )
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
            fig.tight_layout()
            s1_path = os.path.join(fig_b, "segment_time_breakdown.png")
            fig.savefig(s1_path, dpi=120)
            plt.close(fig)
            lines.append("![](figures/track_b/segment_time_breakdown.png)\n\n")
            lines.append(
                "**Figure S1.** Stacked-style bar of segment times (ms) for **one representative** topology row. "
                "Segments are measured **eager-only** with `cfg.use_graph=False`; totals differ from graph-mode "
                "wall time.\n\n"
            )

        for seg in SEGMENT_ORDER:
            rel = _plot_segment_track_b(
                fig_b,
                seg,
                topology_rows,
                "num_object_points",
                "num_object_points",
                f"segment_{seg}_vs_object_points.png",
            )
            if rel:
                lines.append(f"![](figures/track_b/{os.path.basename(rel)})\n\n")
                lines.append(
                    f"**Track (B) segment `{seg}` vs `num_object_points`.** Eager-only ms summed over substeps; "
                    "one outer `step()` per case.\n\n"
                )
            rel2 = _plot_segment_track_b(
                fig_b, seg, topology_rows, "n_springs", "n_springs", f"segment_{seg}_vs_springs.png"
            )
            if rel2:
                lines.append(f"![](figures/track_b/{os.path.basename(rel2)})\n\n")
                lines.append(
                    f"**Track (B) segment `{seg}` vs `n_springs`.** Same segment definition as above.\n\n"
                )
            rel3 = _plot_segment_track_b(
                fig_b, seg, topology_rows, "n_vertices", "n_vertices", f"segment_{seg}_vs_vertices.png"
            )
            if rel3:
                lines.append(f"![](figures/track_b/{os.path.basename(rel3)})\n\n")
                lines.append(
                    f"**Track (B) segment `{seg}` vs `n_vertices`.** Same segment definition as above.\n\n"
                )

    lines.append("## Track (A) — parallel instances (same case)\n\n")
    trc = timing_row_count if timing_row_count is not None else (
        len(parallel_rows) if parallel_rows else 0
    )
    last_n = last_timed_n
    if parallel_rows and last_n is None:
        last_n = parallel_rows[-1].n_instances

    if not parallel_load_complete:
        if trainers_count == 0:
            lines.append(
                "- **Load status:** not started yet (topology track just finished; parallel loads pending).\n"
            )
        else:
            lines.append(
                f"- **Load status:** in progress — **{trainers_count}** trainer(s) loaded successfully; "
                "OOM ceiling not determined until the next load fails or max parallel is reached.\n"
            )
    else:
        lines.append(
            f"- **Load status:** complete — **{trainers_count}** successful trainer load(s); "
            f"first OOM (or stop) at load index **{oom_at_n}** (1-based).\n"
        )

    lines.append(
        f"- **Timing summary:** **{trc}** timing row(s); **last timed N** = **{last_n}** "
        f"(largest `n` in the table below). `timing_complete` = **{parallel_timing_complete}**.\n"
    )

    if parallel_load_complete and trainers_count > 0 and not parallel_timing_complete:
        if parallel_rows and len(parallel_rows) > 0:
            lines.append(
                "- **Timing status:** **incomplete** — stepping ran out of memory partway through the N sweep; "
                "rows below are **partial** (up to the last N that completed).\n"
            )
        else:
            lines.append(
                "- **Timing status:** **incomplete** — stepping hit OOM before any timing row was recorded.\n"
            )
    elif parallel_load_complete and trainers_count > 0 and parallel_timing_complete:
        lines.append("- **Timing status:** complete (all N from 1 to loaded count were timed).\n")
    elif parallel_load_complete and trainers_count == 0:
        lines.append(
            "- **Timing status:** no parallel timing (zero trainers fit — first load hit OOM).\n"
        )

    if parallel_rows is not None and len(parallel_rows) > 0:
        lines.append(
            "| N instances | total step all (ms) | std | per-instance (ms) |\n|---|---:|---:|---:|\n"
        )
        for r in parallel_rows:
            lines.append(
                f"| {r.n_instances} | {r.step_all_ms:.4f} | {r.step_all_std_ms:.4f} | {r.per_instance_ms:.4f} |\n"
            )
        lines.append("\n")

        ns = [r.n_instances for r in parallel_rows]
        tot = [r.step_all_ms for r in parallel_rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ns, tot, marker="o")
        ax.set_xlabel("N (sequential simulators in one timed round)")
        ax.set_ylabel("Time to step all N once (ms)")
        ax.set_title(f"Track (A): total time vs N — timing sweep ({parallel_case})")
        fig.tight_layout()
        p_a1 = os.path.join(fig_a, "track_a_total_time_vs_n.png")
        fig.savefig(p_a1, dpi=120)
        plt.close(fig)
        lines.append("![](figures/track_a/track_a_total_time_vs_n.png)\n\n")
        lines.append(
            "**Figure A1.** One timed round steps **all** of the first `N` simulators **once each**, "
            "sequentially. **Y** is wall time for that full round. This sweep can stop before the **load** "
            "ceiling if stepping OOMs.\n\n"
        )

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax_a.bar([0], [trainers_count], width=0.5, label="trainers_count")
        ax_a.set_xticks([0])
        ax_a.set_xticklabels(["load"])
        ax_a.set_ylabel("Count")
        ax_a.set_title("Load ceiling (trainers loaded)")
        if last_n is not None:
            ax_b.bar([0], [last_n], width=0.5, color="tab:orange", label="last timed N")
        ax_b.set_xticks([0])
        ax_b.set_xticklabels(["timing"])
        ax_b.set_ylabel("N")
        ax_b.set_title("Timing sweep (last N reached)")
        fig.suptitle(
            f"Track (A): load vs timing — oom_at_n (load)={oom_at_n}, "
            f"timing_complete={parallel_timing_complete}"
        )
        fig.tight_layout()
        p_a2 = os.path.join(fig_a, "track_a_load_ceiling_vs_timing.png")
        fig.savefig(p_a2, dpi=120)
        plt.close(fig)
        lines.append("![](figures/track_a/track_a_load_ceiling_vs_timing.png)\n\n")
        lines.append(
            "**Figure A2.** **Left:** how many trainers **loaded** (`trainers_count`). **Right:** last "
            "**N** that completed a timing row (`last_timed_n`). **Load** OOM (`oom_at_n`) can be **higher** "
            "than the last timed N — two different ceilings (memory at load vs activation during stepping).\n\n"
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ns, [r.per_instance_ms for r in parallel_rows], marker="o", color="tab:green")
        ax.set_xlabel("N")
        ax.set_ylabel("per_instance_ms (total round / N)")
        ax.set_title(f"Track (A): per-instance time vs N ({parallel_case})")
        fig.tight_layout()
        p_a3 = os.path.join(fig_a, "track_a_per_instance_vs_n.png")
        fig.savefig(p_a3, dpi=120)
        plt.close(fig)
        lines.append("![](figures/track_a/track_a_per_instance_vs_n.png)\n\n")
        lines.append(
            "**Figure A3.** **Per-instance** time = `total_round_ms / N` for each row (fairness / saturation).\n\n"
        )

        if segment_profile_enabled:
            for seg in SEGMENT_ORDER:
                rela = _plot_segment_track_a(fig_a, seg, parallel_rows, f"segment_{seg}_vs_n.png")
                if rela:
                    lines.append(f"![](figures/track_a/{os.path.basename(rela)})\n\n")
                    lines.append(
                        f"**Segment plot (Track A): `{seg}`.** Sum of eager segment times across **N** "
                        f"one-step passes in one round (same layout as total time).\n\n"
                    )
    else:
        if parallel_rows is None:
            lines.append(
                "_Parallel timing not started yet (track B only, load in progress, or report written before the N sweep)._ \n\n"
            )
        else:
            lines.append(
                "_No parallel timing rows (zero trainers loaded, or timing produced no completed N)._ \n\n"
            )

    lines.append("## Model size glossary\n\n")
    lines.append(
        "- **Topology:** `n_vertices`, `num_object_points`, `n_springs`, control points, substeps.\n"
    )
    lines.append(
        "- **Outer step mean (ms):** one full `step()` call (all substeps); graph path when `cfg.use_graph`.\n"
    )
    lines.append("- **spring_Y MB:** approximate bytes for stiffness log tensor.\n")
    lines.append("- **Peak alloc MB:** `torch.cuda.max_memory_allocated()` around the timed region.\n\n")

    out_path = os.path.join(output_dir, "REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return out_path
