# PhysTwin spring-mass GPU benchmark report
**Generated:** 2026-03-21T21:41:09.308640+00:00  
**Git:** `27d4c3a`  

## Environment

| Field | Value |
|---|---|
| GPU index | 0 |
| Device name | NVIDIA L4 |
| `get_device_properties().total_memory` | 23570219008 bytes |
| `mem_get_info` total | 23570219008 bytes |
| `mem_get_info` free | 23335141376 bytes |
| Sanity (totals match) | **True** |
| CUDA (torch.version.cuda) | 12.1 |
| PyTorch | 2.4.0+cu121 |
| Warp | 1.11.1 |
| nvidia-smi | `NVIDIA L4, 23034` |

## Run log

- 2026-03-21T21:40:51.061895+00:00  Track (B) topology benchmark complete — writing initial REPORT.md
- 2026-03-21T21:40:55.662697+00:00  Parallel load progress: 1 trainer(s) loaded
- 2026-03-21T21:41:00.538121+00:00  Parallel load progress: 2 trainer(s) loaded
- 2026-03-21T21:41:03.308299+00:00  Parallel load phase complete: 2 trainer(s), oom_at_n=3
- 2026-03-21T21:41:09.308521+00:00  Track (A) timing sweep finished — complete

## Methodology

- **Forward only:** one **outer** physics update per timed interval (see glossary). When `cfg.use_graph` is True, the outer step is a **CUDA graph replay** of the same work as `SpringMassSystemWarp.step()` (`wp.capture_launch(forward_graph)`). When `False`, the outer step is a direct `step()` call.
- **Outer step vs substep:** A **substep** is one index `i` in the inner loop `for i in range(num_substeps)` inside `step()`. One **outer step** is the **entire** call to `step()` — including **all** substeps (and graph capture wraps that full inner loop). Table column **“Outer step mean (ms)”** is one mean wall time for that full outer step.
- **Derived column:** **outer ms per substep** = `outer_step_mean_ms / num_substeps` (average duration per inner substep if work were uniform; no separate substep timer).
- **Collision:** production-like — loaded from case + `cfg` (see per-row flags).
- **Data root:** `./data/different_types`
- **Warmup / repeats:** 5 / 20
- **Primary benchmark path `use_graph`:** **True** (same as `cfg.use_graph` during the run).
- **Segment timing (optional):** CUDA events inside `step()` with **`cfg.use_graph=False`** for that measurement. Segment sums are **eager-only** and are **not** equal to graph-mode wall time; use them to see where work goes inside the eager pipeline.
- **Track (A) load ceiling:** `double_lift_cloth_1` — load independent `InvPhyTrainerWarp` instances until first load-time OOM (`oom_at_n = 3`). This is **not** the same as how many simulators can be **stepped** in one timing sweep.
- **Track (A) timing sweep:** For `n = 1 .. K`, one timed round **sequentially** steps the first `n` trainers once each. Peak activation can OOM before `n` reaches the load ceiling — so **timing rows** can stop **before** `trainers_count`.
- **Track (B) topology:** one row per discovered case; different mesh sizes.

**Note:** fewer than 3 topology points (1 cases); scaling plot is still shown if ≥2.


## Glossary (locked terminology)

| Term | Meaning |
|---|---|
| **Outer step** | One call to `SpringMassSystemWarp.step()` **or** one captured graph launch that wraps the same work — **entire** inner substep loop. |
| **Substep** | Single iteration `i` inside `step()`; there are `num_substeps` per outer step. |
| **Track (A) — load ceiling** | Max trainer instances successfully **constructed** before first load-time OOM: `trainers_count`, `oom_at_n` (1-based index of failing load). |
| **Track (A) — timing sweep** | For `n = 1 .. K`, time one sequential round stepping the first `n` trainers once. `K` ≤ `trainers_count`; if stepping OOMs early, `timing_complete` is false and there are fewer rows than loads. |
| **Per-instance time (Track A)** | `total_round_ms / n` for that timed row (`per_instance_ms`). |
| **Segment time (optional)** | Eager-only breakdown of one outer `step()` into pipeline stages; not equivalent to graph replay wall time. |

## Track (B) — topology / per-case

| case | n_vertices | num_object_points | n_springs | substeps | outer ms per substep (derived) | coll. | **Outer step mean (ms)** — one full `step()` | std (ms) | spring_Y MB (approx) | peak alloc MB |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| double_lift_cloth_1 | 4772 | 4742 | 103975 | 667 | 0.094774 | 1 / cfg.self_collision=True | 63.2141 | 0.0641 | 0.3966 | 28.00 |

![](figures/track_b/segment_time_breakdown.png)

**Figure S1.** Stacked-style bar of segment times (ms) for **one representative** topology row. Segments are measured **eager-only** with `cfg.use_graph=False`; totals differ from graph-mode wall time.

![](figures/track_b/segment_control_points_vs_object_points.png)

**Track (B) segment `control_points` vs `num_object_points`.** Eager-only ms summed over substeps; one outer `step()` per case.

![](figures/track_b/segment_control_points_vs_springs.png)

**Track (B) segment `control_points` vs `n_springs`.** Same segment definition as above.

![](figures/track_b/segment_control_points_vs_vertices.png)

**Track (B) segment `control_points` vs `n_vertices`.** Same segment definition as above.

![](figures/track_b/segment_springs_vs_object_points.png)

**Track (B) segment `springs` vs `num_object_points`.** Eager-only ms summed over substeps; one outer `step()` per case.

![](figures/track_b/segment_springs_vs_springs.png)

**Track (B) segment `springs` vs `n_springs`.** Same segment definition as above.

![](figures/track_b/segment_springs_vs_vertices.png)

**Track (B) segment `springs` vs `n_vertices`.** Same segment definition as above.

![](figures/track_b/segment_velocity_integration_vs_object_points.png)

**Track (B) segment `velocity_integration` vs `num_object_points`.** Eager-only ms summed over substeps; one outer `step()` per case.

![](figures/track_b/segment_velocity_integration_vs_springs.png)

**Track (B) segment `velocity_integration` vs `n_springs`.** Same segment definition as above.

![](figures/track_b/segment_velocity_integration_vs_vertices.png)

**Track (B) segment `velocity_integration` vs `n_vertices`.** Same segment definition as above.

![](figures/track_b/segment_object_pair_collision_vs_object_points.png)

**Track (B) segment `object_pair_collision` vs `num_object_points`.** Eager-only ms summed over substeps; one outer `step()` per case.

![](figures/track_b/segment_object_pair_collision_vs_springs.png)

**Track (B) segment `object_pair_collision` vs `n_springs`.** Same segment definition as above.

![](figures/track_b/segment_object_pair_collision_vs_vertices.png)

**Track (B) segment `object_pair_collision` vs `n_vertices`.** Same segment definition as above.

![](figures/track_b/segment_integrate_ground_vs_object_points.png)

**Track (B) segment `integrate_ground` vs `num_object_points`.** Eager-only ms summed over substeps; one outer `step()` per case.

![](figures/track_b/segment_integrate_ground_vs_springs.png)

**Track (B) segment `integrate_ground` vs `n_springs`.** Same segment definition as above.

![](figures/track_b/segment_integrate_ground_vs_vertices.png)

**Track (B) segment `integrate_ground` vs `n_vertices`.** Same segment definition as above.

## Track (A) — parallel instances (same case)

- **Load status:** complete — **2** successful trainer load(s); first OOM (or stop) at load index **3** (1-based).
- **Timing summary:** **2** timing row(s); **last timed N** = **2** (largest `n` in the table below). `timing_complete` = **True**.
- **Timing status:** complete (all N from 1 to loaded count were timed).
| N instances | total step all (ms) | std | per-instance (ms) |
|---|---:|---:|---:|
| 1 | 63.2020 | 0.0494 | 63.2020 |
| 2 | 125.3974 | 0.0462 | 62.6987 |

![](figures/track_a/track_a_total_time_vs_n.png)

**Figure A1.** One timed round steps **all** of the first `N` simulators **once each**, sequentially. **Y** is wall time for that full round. This sweep can stop before the **load** ceiling if stepping OOMs.

![](figures/track_a/track_a_load_ceiling_vs_timing.png)

**Figure A2.** **Left:** how many trainers **loaded** (`trainers_count`). **Right:** last **N** that completed a timing row (`last_timed_n`). **Load** OOM (`oom_at_n`) can be **higher** than the last timed N — two different ceilings (memory at load vs activation during stepping).

![](figures/track_a/track_a_per_instance_vs_n.png)

**Figure A3.** **Per-instance** time = `total_round_ms / N` for each row (fairness / saturation).

![](figures/track_a/segment_control_points_vs_n.png)

**Segment plot (Track A): `control_points`.** Sum of eager segment times across **N** one-step passes in one round (same layout as total time).

![](figures/track_a/segment_springs_vs_n.png)

**Segment plot (Track A): `springs`.** Sum of eager segment times across **N** one-step passes in one round (same layout as total time).

![](figures/track_a/segment_velocity_integration_vs_n.png)

**Segment plot (Track A): `velocity_integration`.** Sum of eager segment times across **N** one-step passes in one round (same layout as total time).

![](figures/track_a/segment_object_pair_collision_vs_n.png)

**Segment plot (Track A): `object_pair_collision`.** Sum of eager segment times across **N** one-step passes in one round (same layout as total time).

![](figures/track_a/segment_integrate_ground_vs_n.png)

**Segment plot (Track A): `integrate_ground`.** Sum of eager segment times across **N** one-step passes in one round (same layout as total time).

## Model size glossary

- **Topology:** `n_vertices`, `num_object_points`, `n_springs`, control points, substeps.
- **Outer step mean (ms):** one full `step()` call (all substeps); graph path when `cfg.use_graph`.
- **spring_Y MB:** approximate bytes for stiffness log tensor.
- **Peak alloc MB:** `torch.cuda.max_memory_allocated()` around the timed region.

