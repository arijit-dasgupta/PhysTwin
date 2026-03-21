# PhysTwin spring-mass GPU benchmark: throughput, scaling, and model size

**Status:** Implemented — **canonical** spec for this benchmark (repo matches §8)  
**Last updated:** 2025-03-21

---

## 0. Plan consolidation (2025-03-21)

- **`spring-mass-performance-benchmark.md`** is **superseded** by **this file** (redirect stub + optional future-work bullets only).
- **Rationale:** One binding spec avoids duplicate open questions and mismatched file paths; the older draft’s scope is largely **realized** here (with explicit non-goals for trainer/backward/multi-GPU v1).

---

## 1. Goal

Deliver **measurable, reproducible** benchmarks for the PhysTwin **Warp** spring-mass simulator:

1. **Model speed:** Wall-clock time per **outer** `step()` (and optionally per **substep**), with CUDA synchronization; derived **throughput** (steps/sec, substeps/sec). **Forward only** — time `SpringMassSystemWarp.step()` with **no backward/autograd** in scope (optional future work only).
2. **Scaling — two tracks in one unified report:**
   - **(A) Parallel instances:** **N independent** `SpringMassSystemWarp` instances on one GPU; measure step time vs N until **first CUDA OOM** (stopping rule).
   - **(B) Topology within one sim:** Scaling of step time with **problem size** — realized using **real PhysTwin cases** of **different** mesh/complexity (one report row per case: `num_object_points`, `n_springs`, etc.). Same report, clearly titled sections.
3. **GPU limit:** Report **maximum N** before first CUDA OOM (parallel track), with methodology documented.
4. **REPORT:** **Single** formatted artifact: **Markdown** with **embedded PNGs** (scaling plots) and **tables**, plus environment and methodology. Document (1) model speed, (2) scaling (parallel + topology), (3) **model size** — topology, parameter tensor footprint, peak memory (see §3).

**Primary code:** `SpringMassSystemWarp` — `qqtt/model/diff_simulator/spring_mass_warp.py`. **Real cases** — same loading path as replay: `rerun_viz/case_setup.py` (`load_case_yaml_and_optimal`, `load_camera_and_intrinsics`), `rerun_viz/replay_core.py` (`load_config_and_camera`, `load_trainer_and_model`), `InvPhyTrainerWarp` — `qqtt/engine/trainer_warp.py`; data under `{base_path}/{case_name}/final_data.pkl`, checkpoints `experiments/{case_name}/train/best_*.pth` (see replay defaults: `--base_path` default `./data/different_types` in `rerun_viz/replay_recorded.py`).

---

## 2. Non-goals

- **Backward / autograd** through `step()` — out of scope unless added later as optional.
- **Full training loop** (optimizer, splatting, wandb, I/O-heavy paths).
- **Multi-node / multi-GPU** unless explicitly requested later.
- **Rerun / visualization** on the timed hot path (no `rerun` logging inside measured loops).

---

## 3. Definitions (locked)

| Term | Meaning |
|------|---------|
| **Outer step** | One call to `SpringMassSystemWarp.step()` = `num_substeps` internal substeps. Report defines this clearly. |
| **Track (A) — parallel** | **N** independent simulators (same real-derived config **or** documented clone of tensors), stepped in one process; increase N until **first `torch.cuda.OutOfMemoryError`** (or Warp/CUDA OOM). Record last successful N−1 as “max stable” and N as OOM boundary (document exact handling). |
| **Track (B) — topology** | **One sim per row**, each row a **different real PhysTwin case** (different `final_data.pkl` / mesh), loaded like replay. Step time vs **num_object_points**, **n_springs**, etc. If only one case size exists in the tree, report that honestly (single row) and still deliver track (A). |
| **Collision — production-like** | Load yaml + case via `load_case_yaml_and_optimal` / trainer as replay does. Simulator gets `self_collision=cfg.self_collision` and masks from data; **`object_collision_flag`** is set inside `SpringMassSystemWarp` when masks imply multi-region collision or `self_collision` is true (`spring_mass_warp.py` ~662–676). **Default: collision on** — do not turn collisions off for the benchmark; **document** `cfg.self_collision`, whether `object_collision_flag == 1`, and `collision_dist` after load. |
| **Model size** | **Topology:** `n_vertices`, `num_object_points`, `n_springs`, `num_control_points`, `num_substeps`, collision flags. **Parameters:** major tensors (`spring_Y`, `rest_lengths`, masses, masks) — element counts and bytes where practical. **Memory:** `torch.cuda.max_memory_allocated` reset/peak around benchmark; optional `torch.cuda.mem_get_info()` for free/total. |
| **GPU environment** | **Current device:** record `torch.cuda.get_device_name()`, index, and **sanity check** — e.g. compare total memory to `torch.cuda.get_device_properties(0).total_memory` and/or `nvidia-smi` parse if available; **fail fast or warn** if printed summary is inconsistent. |

---

## 4. Resolved decisions (formerly open questions)

| # | Decision |
|---|----------|
| 1 | **Both (A) and (B)** in the **same unified report** with **clear section headings** (parallel scaling vs topology-by-case). |
| 2 | **Actual PhysTwin data** — follow **replay** flow: `base_path` + `case_name`, `final_data.pkl`, `experiments/<case>/train/best_*.pth`, yaml/optimal via `rerun_viz/case_setup` + `InvPhyTrainerWarp` as in `replay_core.load_trainer_and_model`. Discover cases under `base_path` (default align with replay) that have required artifacts; allow CLI to filter cases. |
| 3 | **Forward `step()` only** — no backward benchmark in v1. |
| 4 | **Collision on (production-like)** — use loaded cfg + data; **document** effective flags and `object_collision_flag`. |
| 5 | **Report current GPU** — name, memory; **verify** printed info in code (consistency check). |
| 6 | **Parallel strategy** — Implementer chooses **whatever runs without crashing** and **actually executes multiple sims** (sequential stepping in loop, streams, etc.); **document the chosen approach** in the report (and code comments). |
| 7 | **Max parallel:** stop at **first CUDA OOM** (incremental N or binary search — Implementer’s choice; document). |
| 8 | **One report:** Markdown + **embedded PNGs** + tables; all methodology and results in that artifact. Optional CSV/JSON sidecar for replotting — nice-to-have, not required. |
| 9 | **Smoke tests** (tiny, fast, CI-safe) **and** document **long-run** / full CLI usage so manual verification is possible. |
| 10 | **Entry point:** `scripts/bench_spring_mass_gpu.py` (or adjacent name) — **not** `python -m` as primary. |

---

## 5. Plausible codebase touchpoints (Implementer verifies paths)

| Area | Path / note |
|------|-------------|
| Simulator | `qqtt/model/diff_simulator/spring_mass_warp.py` |
| Trainer / load | `qqtt/engine/trainer_warp.py` — `InvPhyTrainerWarp(..., pure_inference_mode=True)` for inference-style sim |
| Replay (reference) | `rerun_viz/replay_recorded.py`, `rerun_viz/replay_core.py`, `rerun_viz/case_setup.py` |
| Config | `qqtt.utils.cfg` |
| Docs | `docs/SPRING_MASS_AND_DATA_STRUCTURES.md`, `data_spec_minimal.txt` |

**Caveat:** Warp device / global state — document constraints; use **`torch.cuda.synchronize()`** + warmup for timings.

---

## 6. Metrics (report columns)

| Metric | Description |
|--------|-------------|
| `step_time_ms` | Mean (optional std/p95) for one `step()` |
| `substep_time_ms` | Optional: `step_time_ms / num_substeps` |
| `steps_per_sec` | Inverse |
| `n_instances` | Track (A) only |
| `case_name` | Track (B) |
| `num_object_points`, `n_springs`, `num_substeps` | Both tracks as applicable |
| `memory_peak_mb` | Peak allocated during benchmark region |
| `collision_documented` | `self_collision`, `object_collision_flag`, `collision_dist` |
| `gpu_info_verified` | Echo sanity-check outcome |

---

## 7. Phased implementation (checklists)

### Phase 0 — Scope lock-in

- [x] Confirm implementation matches locked §3–§4 (no drift); default `base_path` / case discovery aligned with `replay_recorded.py` / `replay_core.py`.

### Phase 1 — Harness + environment

- [x] Add `scripts/bench_spring_mass_gpu.py` (argparse: `--base_path`, `--cases`, `--output_dir`, repeats, warmup, etc.).
- [x] Implement **GPU info** block: `get_device_name`, memory, **sanity check** vs `get_device_properties` (and optional `nvidia-smi`).
- [x] Shared helpers module under `benchmarks/spring_mass_gpu/`: **warmup + timed loop** with `torch.cuda.synchronize()`.
- [x] **Load real case → trainer → simulator** reusing patterns from `load_trainer_and_model` (checkpoint, `set_spring_Y`, `set_collide`, …).

### Phase 2 — Track (B): topology / per-case timings

- [x] For each selected case: record **model size** + mean `step_time_ms` over **forward `step()` only**.
- [x] **≥3 distinct topology points** when ≥3 cases exist; otherwise document limited data.
- [x] **PNG:** e.g. step time vs `n_springs` or `num_object_points` (embedded in Markdown).

### Phase 3 — Track (A): parallel instances until OOM

- [x] Clone **N** simulators from one reference case (or document tensor deep-copy / re-init strategy).
- [x] Step **all** instances per outer iteration (method documented); measure **total** or **per-step** time vs N.
- [x] Increase N until **first CUDA OOM**; record boundary.
- [x] **PNG:** step time or throughput vs N.

### Phase 4 — Single formatted report

- [x] Generate **one** Markdown file under `output_dir` with **relative** `![...](...)` PNG links, **tables**, Environment, Methodology, Results, Limitations.
- [x] Stamp: git commit, Python, `torch`, `warp`, CUDA, GPU name.

### Phase 5 — Tests

- [x] **Smoke:** pytest module that mocks or uses **minimal** GPU path (or skips if no CUDA in CI — document `pytest.mark` / env var).
- [x] **Docs in script `--help`:** how to run **long** / full benchmark locally.

---

## 8. File inventory (as implemented)

| Item | Purpose |
|------|---------|
| `scripts/bench_spring_mass_gpu.py` | CLI entry (primary) |
| `benchmarks/spring_mass_gpu/__init__.py` | Package marker |
| `benchmarks/spring_mass_gpu/timing.py` | Warmup, sync, stats |
| `benchmarks/spring_mass_gpu/load_case.py` | Replay-style trainer + sim load |
| `benchmarks/spring_mass_gpu/discover.py` | Case discovery under `base_path` |
| `benchmarks/spring_mass_gpu/topology.py` | Track (B) per-case stepping |
| `benchmarks/spring_mass_gpu/parallel.py` | Track (A) N-instance allocation + step loop |
| `benchmarks/spring_mass_gpu/sizing.py` | Topology + parameter + memory |
| `benchmarks/spring_mass_gpu/physics.py` | Forward-step helper aligned with replay (no Rerun on hot path) |
| `benchmarks/spring_mass_gpu/gpu_info.py` | Device name, memory, sanity checks |
| `benchmarks/spring_mass_gpu/report.py` | Markdown + matplotlib PNGs |
| `tests/test_spring_mass_gpu_benchmark_smoke.py` | Fast smoke |

---

## 9. Testing & verification (expected)

```bash
pytest tests/test_spring_mass_gpu_benchmark_smoke.py -q
ruff check benchmarks/ scripts/bench_spring_mass_gpu.py

# Full benchmark (manual, long) — example; see script --help for flags
python scripts/bench_spring_mass_gpu.py --base_path ./data/different_types --output_dir benchmarks/reports/run_001
```

---

## 10. Acceptance criteria (checklist)

- [x] **(1) Model speed:** Mean `step_time_ms` for **forward `step()`** on **at least one real case**, with collision settings documented.
- [x] **(2) Scaling (both sections):** Track (A) **step time vs N** up to **OOM**; Track (B) **≥3 topology points** when data allows, else documented limitation.
- [x] **(3) OOM boundary:** **First CUDA OOM** methodology documented; max N or last good N explicit.
- [x] **(4) Model size:** Per row — topology, parameter footprint, memory peak; glossary in report.
- [x] **(5) One report:** Markdown + **embedded PNGs** + tables + **GPU sanity check** + repro instructions.
- [x] **(6) Smoke test** + documented **full** run path.

---

## 11. Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Warp/torch **global state** | Document; subprocess per sweep if needed |
| R2 | **OOM** kills run | Try/except; log N; optional binary search |
| R3 | **Few case sizes** for track (B) | Report honestly; parallel track still delivers value |
| R4 | **Timer noise** | Warmup, multiple iterations, mean ± std |
| R5 | **Matplotlib** dependency | Add to project deps if not already present |

**Rollback:** Remove `scripts/` entry, `benchmarks/spring_mass_gpu/`, and tests.

---

## 12. Related plans & optional future work

| Document | Role |
|----------|------|
| `spring-mass-performance-benchmark.md` | **Superseded** by this file; redirect only. |
| `spring-mass-benchmark-report-depth.md` | **Delta** for richer REPORT (plots, glossary, Track A dual-ceiling clarity, optional `step()` segment profiling). Does not replace this spec. |

**Optional follow-ups** (not required for “implemented” status here): synthetic-only micro sweeps as a **separate** harness; forward+backward timing; `python -m` mirror entry; multi-GPU — each should be a **new** plan if pursued.

---

## 13. Plan verification (read-only audit)

Use after substantive code changes or before release tagging:

- [ ] File list in §8 matches `benchmarks/spring_mass_gpu/` and `scripts/bench_spring_mass_gpu.py`.
- [ ] `pytest tests/test_spring_mass_gpu_benchmark_smoke.py` passes in a suitable environment (GPU or skip as designed).
- [ ] `scripts/bench_spring_mass_gpu.py --help` documents long-run usage.

*(Assign to **Verifier** or maintainer; not a code deliverable.)*
