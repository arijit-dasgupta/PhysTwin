# PhysTwin spring-mass simulation performance benchmark

> **SUPERSEDED (2025-03-21)** — This document is **retained for history and search only**.  
> **Canonical spec, decisions, and implementation map:** [spring-mass-benchmark-gpu-scaling.md](spring-mass-benchmark-gpu-scaling.md)

**Status:** Superseded  
**Last updated:** 2025-03-21  
**Superseded by:** `spring-mass-benchmark-gpu-scaling.md` (binding)

---

## Why this file is no longer authoritative

This was an **earlier draft** with **open questions** (formerly §4, items 1–10) and overlapping **phased implementation** content. The **GPU scaling** plan locked definitions (tracks A/B, real-case loading, collision defaults, OOM rule, report shape, CLI entry) and matches what exists in the repo: `benchmarks/spring_mass_gpu/`, `scripts/bench_spring_mass_gpu.py`, and `tests/test_spring_mass_gpu_benchmark_smoke.py`.

**Do not** use the old §7 phased checklists or §10 acceptance criteria for current work—follow the canonical plan.

---

## Optional future work (not in current GPU benchmark scope)

Ideas from this draft that **may** become a **separate** follow-up plan if needed:

- **Synthetic-only micro sweeps** (controlled `n_vertices` / `n_springs` within one sim) as a **primary** scaling curve—the implemented benchmark emphasizes **real PhysTwin cases** for topology track (B) per the canonical spec.
- **Forward + backward** timing or **full training-loop** benchmarking (explicit non-goals in the canonical plan for v1).
- **Multi-node / multi-GPU** scaling.

---

## Archive note

Original sections (goal, assumptions, metrics tables, risks, expected file paths like `benchmarks/spring_mass/`) described a **broader** benchmark harness. The **implemented** layout is `benchmarks/spring_mass_gpu/*` and `scripts/bench_spring_mass_gpu.py` as documented in the canonical plan §8.
