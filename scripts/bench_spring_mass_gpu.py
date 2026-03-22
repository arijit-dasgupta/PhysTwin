#!/usr/bin/env python3
"""
PhysTwin spring-mass GPU benchmark (forward step only).

Loads real cases like ``rerun_viz.replay_core``; measures per-case topology (track B)
and parallel simulator instances until OOM (track A). Writes Markdown reports and PNGs.

**Incremental saves:** After track (B) completes, a first ``REPORT.md`` is written (with B figures).
During track (A) loading, the report is rewritten after **each** successful trainer load so a crash
or OOM still leaves topology data and load progress on disk. A final write runs in ``finally`` so
even unexpected failures append a **Run failure** section with traceback.

**OOM on load is expected** for track (A): the loader probes how many independent trainers fit until
CUDA/Warp runs out of memory. That is normal behavior, not a benchmark bug.

Examples::

  # Full run (requires GPU + data + checkpoints)
  python scripts/bench_spring_mass_gpu.py --output_dir benchmarks/reports/run_001

  # Subset of cases
  python scripts/bench_spring_mass_gpu.py --output_dir /tmp/b --cases double_lift_cloth_3

Smoke / CI: ``pytest tests/test_spring_mass_gpu_benchmark_smoke.py``
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from datetime import datetime, timezone


def main() -> None:
    import torch

    from benchmarks.spring_mass_gpu.discover import discover_cases
    from benchmarks.spring_mass_gpu.gpu_info import collect_gpu_report
    from benchmarks.spring_mass_gpu.parallel import rows_for_all_n, trainers_until_oom
    from benchmarks.spring_mass_gpu.report import write_report
    from benchmarks.spring_mass_gpu.topology import benchmark_one_case
    from qqtt.utils import cfg

    _epilog = """\
Full / long local run (all discoverable cases under --base_path; requires CUDA, final_data.pkl and
experiments/<case>/train/best_*.pth per case):

  python scripts/bench_spring_mass_gpu.py --output_dir benchmarks/reports/run_001

Reports are written incrementally (after topology, after each parallel trainer load, and in a final
finally block) so you keep partial results if the process OOMs or crashes. OOM while loading extra
trainers in track (A) is expected and defines the parallel ceiling.

Optional: --cases a,b to subset; --warmup / --repeats / --max_parallel tune measurement load.

Smoke / CI: pytest tests/test_spring_mass_gpu_benchmark_smoke.py
"""
    parser = argparse.ArgumentParser(
        description="Spring-mass GPU benchmark (PhysTwin real data).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog,
    )
    parser.add_argument(
        "--base_path",
        default="./data/different_types",
        help="Dataset root (same as replay).",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for REPORT.md and figures/")
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case names (default: all discoverable).",
    )
    parser.add_argument(
        "--parallel_case",
        default=None,
        help="Case name for track (A) parallel loads (default: first discovered case).",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max_parallel", type=int, default=128)
    parser.set_defaults(segment_profile=False)
    parser.add_argument(
        "--segment-profile",
        dest="segment_profile",
        action="store_true",
        help="Run eager CUDA segment profiling (extra overhead; richer report).",
    )
    parser.add_argument(
        "--no-segment-profile",
        dest="segment_profile",
        action="store_false",
        help="Disable segment profiling (default).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)

    gpu = collect_gpu_report()
    if not gpu.sanity_total_mem_match:
        print(
            "[warn] GPU memory sanity check: get_device_properties vs mem_get_info mismatch.",
            file=sys.stderr,
        )

    cases = discover_cases(args.base_path)
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c in want]
    if not cases:
        print(
            f"No cases found under {args.base_path!r} with final_data.pkl + experiments/.../best_*.pth.",
            file=sys.stderr,
        )
        sys.exit(3)

    parallel_case = (args.parallel_case or cases[0]).strip()
    if parallel_case not in cases:
        parallel_case = cases[0]

    def ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    event_log: list[str] = []
    error_summary: str | None = None
    topology_rows: list = []
    trainers: list = []
    oom_at: int | None = None
    parallel_rows: list | None = None
    parallel_timing_complete = False
    parallel_load_complete = False
    notes = ""
    out_path = ""

    def log_evt(msg: str) -> None:
        event_log.append(f"{ts()}  {msg}")

    def emit_report() -> str:
        nonlocal out_path
        trows = parallel_rows or []
        last_n = trows[-1].n_instances if trows else None
        out_path = write_report(
            args.output_dir,
            gpu=gpu,
            topology_rows=topology_rows,
            parallel_rows=parallel_rows,
            trainers_count=len(trainers),
            oom_at_n=oom_at,
            parallel_case=parallel_case,
            base_path=args.base_path,
            warmup=args.warmup,
            repeats=args.repeats,
            extra_notes=notes,
            event_log=list(event_log),
            error_summary=error_summary,
            parallel_timing_complete=parallel_timing_complete,
            parallel_load_complete=parallel_load_complete,
            timing_row_count=len(trows),
            last_timed_n=last_n,
            segment_profile_enabled=args.segment_profile,
            benchmark_use_graph=bool(cfg.use_graph),
        )
        return out_path

    try:
        for cn in cases:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            topology_rows.append(
                benchmark_one_case(
                    args.base_path,
                    cn,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    segment_profile=args.segment_profile,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()

        if len(topology_rows) < 3:
            notes += f"**Note:** fewer than 3 topology points ({len(topology_rows)} cases); scaling plot is still shown if ≥2.\n"

        log_evt("Track (B) topology benchmark complete — writing initial REPORT.md")
        parallel_rows = None
        parallel_timing_complete = False
        parallel_load_complete = False
        trainers = []
        oom_at = None
        emit_report()

        def on_progress(tr_list: list) -> None:
            nonlocal trainers
            trainers = tr_list
            log_evt(f"Parallel load progress: {len(trainers)} trainer(s) loaded")
            emit_report()

        trainers, oom_at = trainers_until_oom(
            args.base_path,
            parallel_case,
            max_instances=args.max_parallel,
            on_progress=on_progress,
        )
        parallel_load_complete = True
        log_evt(f"Parallel load phase complete: {len(trainers)} trainer(s), oom_at_n={oom_at}")
        emit_report()

        w_tm = max(1, min(3, args.warmup))
        r_tm = max(1, min(10, args.repeats))
        if trainers:
            parallel_rows, parallel_timing_complete = rows_for_all_n(
                trainers,
                warmup=w_tm,
                repeats=r_tm,
                segment_profile=args.segment_profile,
            )
        else:
            parallel_rows = []
            parallel_timing_complete = True

        log_evt(
            "Track (A) timing sweep finished — "
            + ("complete" if parallel_timing_complete else "partial (OOM during timing)")
        )

    except Exception:
        error_summary = traceback.format_exc()
        log_evt("Uncaught exception — final report will include traceback")
        raise
    finally:
        try:
            out_path = emit_report()
            print(out_path)
        except Exception as ex:
            print(f"[error] write_report in finally failed: {ex}", file=sys.stderr)


if __name__ == "__main__":
    main()
