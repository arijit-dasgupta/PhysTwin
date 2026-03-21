# PhysTwin spring-mass GPU benchmark report
**Generated:** 2026-03-21T21:50:03.556576+00:00  
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

- 2026-03-21T21:43:46.914562+00:00  Track (B) topology benchmark complete — writing initial REPORT.md
- 2026-03-21T21:43:53.942458+00:00  Parallel load progress: 1 trainer(s) loaded
- 2026-03-21T21:44:00.861455+00:00  Parallel load progress: 2 trainer(s) loaded
- 2026-03-21T21:44:07.811759+00:00  Parallel load progress: 3 trainer(s) loaded
- 2026-03-21T21:44:14.722750+00:00  Parallel load progress: 4 trainer(s) loaded
- 2026-03-21T21:44:21.610853+00:00  Parallel load progress: 5 trainer(s) loaded
- 2026-03-21T21:44:28.172289+00:00  Parallel load progress: 6 trainer(s) loaded
- 2026-03-21T21:44:35.081554+00:00  Parallel load progress: 7 trainer(s) loaded
- 2026-03-21T21:44:42.016483+00:00  Parallel load progress: 8 trainer(s) loaded
- 2026-03-21T21:44:48.927555+00:00  Parallel load progress: 9 trainer(s) loaded
- 2026-03-21T21:44:55.886892+00:00  Parallel load progress: 10 trainer(s) loaded
- 2026-03-21T21:45:02.872640+00:00  Parallel load progress: 11 trainer(s) loaded
- 2026-03-21T21:45:09.826947+00:00  Parallel load progress: 12 trainer(s) loaded
- 2026-03-21T21:45:16.369869+00:00  Parallel load progress: 13 trainer(s) loaded
- 2026-03-21T21:45:23.348536+00:00  Parallel load progress: 14 trainer(s) loaded
- 2026-03-21T21:45:30.326398+00:00  Parallel load progress: 15 trainer(s) loaded
- 2026-03-21T21:45:37.337946+00:00  Parallel load progress: 16 trainer(s) loaded
- 2026-03-21T21:45:43.907610+00:00  Parallel load progress: 17 trainer(s) loaded
- 2026-03-21T21:45:50.917317+00:00  Parallel load progress: 18 trainer(s) loaded
- 2026-03-21T21:45:57.942648+00:00  Parallel load progress: 19 trainer(s) loaded
- 2026-03-21T21:46:04.536363+00:00  Parallel load progress: 20 trainer(s) loaded
- 2026-03-21T21:46:11.573686+00:00  Parallel load progress: 21 trainer(s) loaded
- 2026-03-21T21:46:18.644945+00:00  Parallel load progress: 22 trainer(s) loaded
- 2026-03-21T21:46:25.206726+00:00  Parallel load progress: 23 trainer(s) loaded
- 2026-03-21T21:46:32.267480+00:00  Parallel load progress: 24 trainer(s) loaded
- 2026-03-21T21:46:38.841710+00:00  Parallel load progress: 25 trainer(s) loaded
- 2026-03-21T21:46:45.935952+00:00  Parallel load progress: 26 trainer(s) loaded
- 2026-03-21T21:46:53.096772+00:00  Parallel load progress: 27 trainer(s) loaded
- 2026-03-21T21:46:59.755164+00:00  Parallel load progress: 28 trainer(s) loaded
- 2026-03-21T21:47:06.904890+00:00  Parallel load progress: 29 trainer(s) loaded
- 2026-03-21T21:47:13.463992+00:00  Parallel load progress: 30 trainer(s) loaded
- 2026-03-21T21:47:20.634271+00:00  Parallel load progress: 31 trainer(s) loaded
- 2026-03-21T21:47:27.207024+00:00  Parallel load progress: 32 trainer(s) loaded
- 2026-03-21T21:47:34.324935+00:00  Parallel load progress: 33 trainer(s) loaded
- 2026-03-21T21:47:40.886636+00:00  Parallel load progress: 34 trainer(s) loaded
- 2026-03-21T21:47:48.064934+00:00  Parallel load progress: 35 trainer(s) loaded
- 2026-03-21T21:47:54.664591+00:00  Parallel load progress: 36 trainer(s) loaded
- 2026-03-21T21:48:01.879256+00:00  Parallel load progress: 37 trainer(s) loaded
- 2026-03-21T21:48:08.441982+00:00  Parallel load progress: 38 trainer(s) loaded
- 2026-03-21T21:48:15.639589+00:00  Parallel load progress: 39 trainer(s) loaded
- 2026-03-21T21:48:22.174786+00:00  Parallel load progress: 40 trainer(s) loaded
- 2026-03-21T21:48:29.372020+00:00  Parallel load progress: 41 trainer(s) loaded
- 2026-03-21T21:48:35.917163+00:00  Parallel load progress: 42 trainer(s) loaded
- 2026-03-21T21:48:43.209594+00:00  Parallel load progress: 43 trainer(s) loaded
- 2026-03-21T21:48:49.489805+00:00  Parallel load phase complete: 43 trainer(s), oom_at_n=44
- 2026-03-21T21:50:03.556441+00:00  Track (A) timing sweep finished — partial (OOM during timing)

## Methodology

- **Forward only:** one **outer** physics update per timed interval (see glossary). When `cfg.use_graph` is True, the outer step is a **CUDA graph replay** of the same work as `SpringMassSystemWarp.step()` (`wp.capture_launch(forward_graph)`). When `False`, the outer step is a direct `step()` call.
- **Outer step vs substep:** A **substep** is one index `i` in the inner loop `for i in range(num_substeps)` inside `step()`. One **outer step** is the **entire** call to `step()` — including **all** substeps (and graph capture wraps that full inner loop). Table column **“Outer step mean (ms)”** is one mean wall time for that full outer step.
- **Derived column:** **outer ms per substep** = `outer_step_mean_ms / num_substeps` (average duration per inner substep if work were uniform; no separate substep timer).
- **Collision:** production-like — loaded from case + `cfg` (see per-row flags).
- **Data root:** `./data/different_types`
- **Warmup / repeats:** 5 / 20
- **Primary benchmark path `use_graph`:** **True** (same as `cfg.use_graph` during the run).
- **Segment timing (optional):** CUDA events inside `step()` with **`cfg.use_graph=False`** for that measurement. Segment sums are **eager-only** and are **not** equal to graph-mode wall time; use them to see where work goes inside the eager pipeline.
- **Track (A) load ceiling:** `double_lift_cloth_1` — load independent `InvPhyTrainerWarp` instances until first load-time OOM (`oom_at_n = 44`). This is **not** the same as how many simulators can be **stepped** in one timing sweep.
- **Track (A) timing sweep:** For `n = 1 .. K`, one timed round **sequentially** steps the first `n` trainers once each. Peak activation can OOM before `n` reaches the load ceiling — so **timing rows** can stop **before** `trainers_count`.
- **Track (B) topology:** one row per discovered case; different mesh sizes.

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
| double_lift_cloth_1 | 4772 | 4742 | 103975 | 667 | 0.094855 | 1 / cfg.self_collision=True | 63.2683 | 0.0443 | 0.3966 | 28.00 |
| double_lift_cloth_3 | 7551 | 7521 | 37566 | 667 | 0.085438 | 1 / cfg.self_collision=True | 56.9872 | 0.4140 | 0.1433 | 40.65 |
| double_lift_sloth | 6690 | 6660 | 114997 | 667 | 0.056114 | 1 / cfg.self_collision=True | 37.4282 | 0.0843 | 0.4387 | 17.75 |
| double_lift_zebra | 4637 | 4607 | 74893 | 667 | 0.053335 | 1 / cfg.self_collision=True | 35.5742 | 0.1476 | 0.2857 | 12.71 |
| double_stretch_sloth | 6517 | 6487 | 53166 | 667 | 0.129776 | 1 / cfg.self_collision=True | 86.5603 | 0.2250 | 0.2028 | 44.26 |
| double_stretch_zebra | 4238 | 4208 | 79982 | 667 | 0.188546 | 1 / cfg.self_collision=True | 125.7603 | 0.3539 | 0.3051 | 33.35 |
| rope_double_hand | 1783 | 1753 | 13729 | 667 | 0.371648 | 1 / cfg.self_collision=True | 247.8894 | 0.0664 | 0.0524 | 15.11 |
| single_clift_cloth_1 | 6520 | 6490 | 108508 | 667 | 0.098970 | 1 / cfg.self_collision=True | 66.0127 | 0.3829 | 0.4139 | 27.06 |
| single_clift_cloth_3 | 8188 | 8158 | 40707 | 667 | 0.065972 | 1 / cfg.self_collision=True | 44.0034 | 0.2151 | 0.1553 | 36.57 |
| single_lift_cloth | 7823 | 7793 | 105686 | 667 | 0.297581 | 1 / cfg.self_collision=True | 198.4866 | 0.4200 | 0.4032 | 64.95 |
| single_lift_cloth_1 | 5520 | 5490 | 36158 | 667 | 0.051119 | 1 / cfg.self_collision=True | 34.0966 | 0.2292 | 0.1379 | 28.06 |
| single_lift_cloth_3 | 8612 | 8582 | 61361 | 667 | 0.035836 | 1 / cfg.self_collision=True | 23.9027 | 0.0817 | 0.2341 | 63.37 |
| single_lift_cloth_4 | 8215 | 8185 | 118812 | 667 | 0.116961 | 1 / cfg.self_collision=True | 78.0127 | 0.1210 | 0.4532 | 67.41 |
| single_lift_dinosor | 7606 | 7576 | 118314 | 667 | 0.132051 | 1 / cfg.self_collision=True | 88.0779 | 0.4834 | 0.4513 | 23.13 |
| single_lift_rope | 2178 | 2148 | 27655 | 667 | 0.328809 | 1 / cfg.self_collision=True | 219.3158 | 0.0320 | 0.1055 | 4.66 |
| single_lift_sloth | 6925 | 6895 | 110833 | 667 | 0.138732 | 1 / cfg.self_collision=True | 92.5343 | 0.2575 | 0.4228 | 23.56 |
| single_lift_zebra | 4964 | 4934 | 54989 | 667 | 0.137502 | 1 / cfg.self_collision=True | 91.7141 | 0.2285 | 0.2098 | 13.80 |
| single_push_rope | 2362 | 2332 | 38557 | 667 | 0.163999 | 1 / cfg.self_collision=True | 109.3875 | 0.1039 | 0.1471 | 5.53 |
| single_push_rope_1 | 2323 | 2293 | 45498 | 667 | 0.210654 | 1 / cfg.self_collision=True | 140.5064 | 0.1913 | 0.1736 | 8.70 |
| single_push_rope_4 | 885 | 855 | 18149 | 667 | 0.043785 | 1 / cfg.self_collision=True | 29.2048 | 0.2422 | 0.0692 | 3.77 |
| single_push_sloth | 6300 | 6270 | 33599 | 667 | 0.126189 | 1 / cfg.self_collision=True | 84.1679 | 0.4675 | 0.1282 | 15.88 |
| weird_package | 4562 | 4532 | 44256 | 667 | 0.462090 | 1 / cfg.self_collision=True | 308.2142 | 0.0905 | 0.1688 | 9.53 |

![Track B1 — springs](figures/track_b/track_b_time_vs_springs.png)

**Figure B1.** Each point is one real case. **Y** is **outer step mean (ms)** — one full `step()` including all substeps (graph path when `use_graph` is enabled for the benchmark). **X** is `n_springs`.

![Track B2 — object points](figures/track_b/track_b_time_vs_object_points.png)

**Figure B2.** Same outer-step mean as B1. **X** is `num_object_points` (mesh points on the object).

![Track B3 — vertices](figures/track_b/track_b_time_vs_vertices.png)

**Figure B3.** Same outer-step mean. **X** is `n_vertices` (full vertex count).

![Track B4 — loglog](figures/track_b/track_b_time_vs_springs_loglog.png)

**Figure B4 (optional).** Log–log view of B1 for rough scaling intuition.

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

- **Load status:** complete — **43** successful trainer load(s); first OOM (or stop) at load index **44** (1-based).
- **Timing summary:** **13** timing row(s); **last timed N** = **13** (largest `n` in the table below). `timing_complete` = **False**.
- **Timing status:** **incomplete** — stepping ran out of memory partway through the N sweep; rows below are **partial** (up to the last N that completed).
| N instances | total step all (ms) | std | per-instance (ms) |
|---|---:|---:|---:|
| 1 | 33.4596 | 1.3330 | 33.4596 |
| 2 | 66.5089 | 2.5122 | 33.2544 |
| 3 | 101.6420 | 2.3615 | 33.8807 |
| 4 | 130.2980 | 4.4604 | 32.5745 |
| 5 | 162.4225 | 4.5758 | 32.4845 |
| 6 | 193.9623 | 5.5874 | 32.3271 |
| 7 | 228.7997 | 7.7641 | 32.6857 |
| 8 | 261.6801 | 5.2496 | 32.7100 |
| 9 | 293.5174 | 8.5901 | 32.6130 |
| 10 | 326.1710 | 7.2140 | 32.6171 |
| 11 | 359.4953 | 8.4406 | 32.6814 |
| 12 | 390.0903 | 8.1851 | 32.5075 |
| 13 | 420.5677 | 4.3309 | 32.3514 |

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

