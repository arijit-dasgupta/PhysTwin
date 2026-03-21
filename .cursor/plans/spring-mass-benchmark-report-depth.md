# PhysTwin spring-mass GPU benchmark — **report depth** + Track (A) clarity + step segmentation

**Status:** Ready for implementer  
**Last updated:** 2025-03-21  
**Parent spec:** `spring-mass-benchmark-gpu-scaling.md` (baseline benchmark harness + tracks — **unchanged** as the structural spec). This document is the **delta** for a **richer REPORT**, optional **fine-grained timing**, and **locked terminology** so readers do not conflate two different “parallel” ceilings.

---

## 1. Goal

Enhance the **Markdown + PNG** output (`benchmarks/spring_mass_gpu/report.py` and data fed into it) so that the artifact is **much more detailed**:

1. **Narrative + structure:** methodology, figure **captions**, a **glossary** (terms used in tables/plots), and room for **many** plots (acceptable to “massively populate” the report).
2. **Model size metrics:** use `ModelSizeRecord` / `SpringMassSystemWarp` fields **`n_vertices`**, **`num_object_points`**, **`n_springs`** (and existing control-point / substep fields where useful) in **tables** and **additional scatter plots** (e.g. mean outer-step time vs `num_object_points`, vs `n_springs`, vs `n_vertices` for Track (B)).
3. **Disambiguate Track (A):** explain two **independent** concepts that users confuse:
   - **(a) Loaded trainer instances:** how many **independent** `InvPhyTrainerWarp` (+ sim state) fit in GPU memory — `trainers_until_oom` → `trainers_count`, `oom_at_n` on **load**.
   - **(b) Timing sweep N:** for each timed round, **N** simulators are **stepped sequentially** in one process (`benchmark_step_k_parallel`); peak **activation** during that round can **OOM at a smaller N** than the load ceiling — so the **timing plot may stop at N = 13** while **43** trainers **loaded** and **`oom_at_n = 44`** (numbers illustrative). Document **two ceilings**; prefer **two figures** and/or **one annotated figure** (e.g. vertical lines, shaded regions, captions).
4. **Substeps vs outer step:** **Substep** = inner loop index `i` in `SpringMassSystemWarp.step()` (`for i in range(self.num_substeps)`). **Reported “step mean (ms)”** = **one outer** `step()` = full physics update including **all** substeps (and full CUDA graph capture when `cfg.use_graph` and replay-style path uses `wp.capture_launch(forward_graph)`). State this **explicitly** in the report. Optional **derived** column: **per-substep average (ms)** = `outer_step_ms / num_substeps` (no separate timer required unless trivial).
5. **Fine-grained step timing (optional / second pass):** instrument **logical segments** inside `SpringMassSystemWarp.step()` (see `qqtt/model/diff_simulator/spring_mass_warp.py` ~972–1061): e.g. **control points** (`set_control_points`), **springs** (`eval_springs`), **velocity from force** (`update_vel_from_force`), **object-collision branch** (`object_collision` when `object_collision_flag`), **integrate + ground** (`integrate_ground_collision`). Group sensibly (few bars, not one bar per micro-launch if noisy). Use **CUDA events** + `torch.cuda.synchronize()` between segments when profiling. **Tradeoff:** when `cfg.use_graph` is True, production/replay uses **`wp.capture_launch(forward_graph)`** — segment splits require the **eager** `step()` path (`cfg.use_graph=False`) for faithful per-segment times, **or** a **benchmark-only** eager duplicate path; document in **Methodology** and **Risks**.

---

## 2. Non-goals

- Changing the **primary** benchmark semantics of Track (A)/(B) from the parent spec unless required for clarity (this work is **report + optional instrumentation**).
- **Backward / autograd** timing (still out of scope).
- **Guaranteeing** segment timings match graph mode wall time (they intentionally profile **eager** decomposition).

---

## 3. Locked terminology (reader-facing)

| Term | Meaning |
|------|---------|
| **Outer step** | One call to `SpringMassSystemWarp.step()` **or** one captured graph launch that wraps the same work — **entire** inner substep loop. |
| **Substep** | Single iteration `i` inside `step()`; there are `num_substeps` per outer step. |
| **Track (A) — load ceiling** | Max **trainer** instances successfully **constructed** before first load-time OOM: `trainers_count`, `oom_at_n` (1-based index of failing load). Memory holds **weights + sim state** for each instance. |
| **Track (A) — timing sweep** | For `n = 1 .. K`, time **one sequential round** stepping the first `n` trainers once each (`rows_for_all_n`). **`K` ≤ `trainers_count`**; if stepping OOMs early, **`timing_complete`** is false and **fewer** rows than loads — **not a bug**. |
| **Per-instance time (Track A)** | `total_round_ms / n` for that timed row (already computed as `per_instance_ms`). |
| **Segment time (optional)** | Eager-only breakdown of **one outer** `step()` into pipeline stages; not the same as a single **substep** unless reported as such. |

---

## 4. File touch list (Implementer verifies)

| Area | Path | Role |
|------|------|------|
| Simulator (segments) | `qqtt/model/diff_simulator/spring_mass_warp.py` | Optional: CUDA event pairs / guarded profiling hooks inside `step()` (or refactor minimal inner loop for reuse). |
| Benchmark package | `benchmarks/spring_mass_gpu/report.py` | Rich Markdown: sections, glossary, many plots, Track (A) two-ceiling explanation, captions. |
| | `benchmarks/spring_mass_gpu/sizing.py` | Already exposes `ModelSizeRecord`; ensure any **new** table columns pull from here consistently. |
| | `benchmarks/spring_mass_gpu/physics.py` | Document interaction with `cfg.use_graph`; if segment benchmark runs **eager**, force/document `use_graph=False` for that phase. |
| | `benchmarks/spring_mass_gpu/parallel.py` | Expose **load count** vs **timing row count** + `timing_complete` to report (already partially there — wire **explicit** narrative fields/labels). |
| | `benchmarks/spring_mass_gpu/topology.py` | If segment timing added for Track (B), pass stats into report rows. |
| Entry | `scripts/bench_spring_mass_gpu.py` | CLI flags: e.g. `--segment_profile`, `--no-graph-for-segments` (names TBD); pass through to report. |
| Tests | `tests/test_spring_mass_gpu_benchmark_smoke.py` | Adjust/extend for new report sections or mocked segment path (keep CI fast). |
| Optional | `tests/test_replay_helpers.py` or new `tests/test_spring_mass_report_*` | Light tests for pure report helpers (if extracted). |

---

## 5. Plot inventory (names + axes + caption intent)

Implementer may adjust **filenames** for consistency; keep **stable** relative paths under `figures/` for Markdown.

| ID | Suggested filename | Axes / content | Caption intent |
|----|--------------------|----------------|----------------|
| B1 | `track_b_time_vs_springs.png` | x: `n_springs`, y: mean outer-step ms | Baseline scaling with spring count (may replace/keep existing). |
| B2 | `track_b_time_vs_object_points.png` | x: `num_object_points`, y: mean outer-step ms | Topology scaling by **object points** (user-requested). |
| B3 | `track_b_time_vs_vertices.png` | x: `n_vertices`, y: mean outer-step ms | Full mesh vertex count vs time. |
| B4 | *(optional)* `track_b_time_vs_springs_loglog.png` | log–log same as B1 | Rough **scaling exponent** intuition (optional). |
| A1 | `track_a_total_time_vs_n.png` | x: N timed, y: ms for one full sequential pass over N sims | **Timing sweep** — may truncate before load ceiling. |
| A2 | `track_a_load_ceiling_vs_timing.png` | **Composite or dual-panel:** (left) load probe outcome; (right) timing sweep — or single plot with **annotations** | **Disambiguate** `trainers_count` vs **last N** timed; show `oom_at_n` for **load**. |
| A3 | *(optional)* `track_a_per_instance_vs_n.png` | x: N, y: `per_instance_ms` | Fairness / saturation narrative. |
| S1 | *(optional)* `segment_time_breakdown.png` | x: segment labels, y: ms (or **stacked 100%**) | **Eager-only** pipeline breakdown for **one representative case**; caption **must** state `use_graph=False` for this measurement. |

**Tables:** Track (B) should add **`n_vertices`**, optional **per-substep avg ms** column; Track (A) table should add a **footnote or adjacent summary** row: **loaded trainers**, **`oom_at_n`**, **timing rows completed**, **`timing_complete`**.

---

## 6. Phased implementation (`- [ ]` checklists)

### Phase R1 — Report narrative + Track (A) clarity + model-size plots

- [ ] Expand **Methodology**: outer step vs substep; sequential stepping for Track (A); two ceilings (load vs timing).
- [ ] Add **Glossary** (or expand existing) with locked terms from §3.
- [ ] Track (B) table: include **`n_vertices`** (and keep `num_object_points`, `n_springs`).
- [ ] Generate **B2**, **B3** (and keep/refine **B1**); ensure captions reference **outer** step ms.
- [ ] Track (A): **two figures or one annotated figure** explaining **why** timing N can be **<** loaded count; reference `parallel_load_complete`, `trainers_count`, `oom_at_n`, `parallel_timing_complete`.
- [ ] Optional derived column: **outer_step_ms / num_substeps**.

### Phase R2 — Fine-grained segment timing (optional)

- [ ] Add **guarded** instrumentation in `spring_mass_warp.py` (no overhead when disabled) **or** benchmark-only wrapper approved in review.
- [ ] Run segment profile with **`cfg.use_graph=False`** (or dedicated code path); document **non-equivalence** to graph wall time.
- [ ] Emit **S1** + methodology paragraph on **graph vs eager** tradeoff.
- [ ] Wire CLI + report section; default **off** if runtime cost is high.

### Phase R3 — Tests & verification

- [ ] Update smoke tests so they still pass (skip GPU as today).
- [ ] `ruff`/pytest for touched modules.

---

## 7. Testing & verification

```bash
pytest tests/test_spring_mass_gpu_benchmark_smoke.py -q
ruff check benchmarks/spring_mass_gpu scripts/bench_spring_mass_gpu.py qqtt/model/diff_simulator/spring_mass_warp.py
# Manual: full run with GPU + data
python scripts/bench_spring_mass_gpu.py --output_dir benchmarks/reports/run_001
```

---

## 8. Acceptance criteria (`- [ ]`)

- [ ] Report **explicitly** defines **outer step**, **substep**, and **Track (A) load ceiling vs timing sweep**; a reader seeing **43 loads** vs **13 timing rows** understands **two different limits**.
- [ ] Track (B) presents **nodes** metrics: **`num_object_points`**, **`n_vertices`**, **`n_springs`** in tables and **at least two** new scatter plots (object points + vertices, or as specified in §5).
- [ ] **“Step mean (ms)”** is clearly **one outer** step (all substeps; graph path when enabled) and optional **per-substep average** is labeled **derived**.
- [ ] If segment timing ships: **methodology** states **eager vs graph**; figure caption states profiling mode.
- [ ] Smoke test passes; no unexplained regression in benchmark entry behavior.

---

## 9. Risks

| # | Risk | Mitigation |
|---|------|------------|
| G1 | **CUDA graph** hides internal phases | Document; segment plots **eager-only**; optional note on total outer-step time from production path. |
| R1 | **Synchronize** overhead inflates segment sums vs single outer timer | Report **sum of segments ≈ outer** check for same config; use **CUDA events** carefully. |
| R2 | **Instrumentation** in core `spring_mass_warp.py` risks merge conflict / perf | Guard with `if self._profile_segments:` or zero-cost default; keep diff minimal. |
| R3 | **Track (A) OOM during timing** partial rows | Already returns `timing_complete`; report must **prominently** mark partial vs complete. |
| R4 | **Report size** (many PNGs) | Acceptable per user; consider **subfolders** under `figures/` if needed (`track_a/`, `track_b/`). |

---

## 10. Open questions for the user

*(None blocking — Implementer may choose figure layout, optional log–log plot, and exact CLI flag names within §4–§6.)*

---

READY FOR IMPLEMENTER

**Summary:** Add a **sibling** benchmark-report initiative: enrich `report.py` (and wiring from `parallel.py` / topology) with **glossary**, **methodology**, **Track (A) dual-ceiling** figures and text, **Track (B)** tables/plots including **`n_vertices`** and **`num_object_points`**, optional **per-substep derived ms**, and **optional eager-only** `step()` **segment profiling** via CUDA events in `spring_mass_warp.py` with **documented graph vs eager** tradeoffs. **Suggested first phase:** Phase **R1** (no simulator instrumentation yet) so the report stands alone; Phase **R2** adds segments behind a flag.

**Plan file:** `.cursor/plans/spring-mass-benchmark-report-depth.md`  
**Suggested first phase:** **R1** — narrative, glossary, Track (A) annotated figures, extra Track (B) scatter plots and table columns.
