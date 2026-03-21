# PhysTwin spring-mass GPU benchmark report
**Generated:** 2026-03-21T21:36:54.723108+00:00  
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

- 2026-03-21T21:36:54.722923+00:00  Uncaught exception — final report will include traceback

## Run failure

The benchmark exited with an error. Details:

```
Traceback (most recent call last):
  File "/home/arijitdasgupta/PhysTwin/scripts/bench_spring_mass_gpu.py", line 180, in main
    benchmark_one_case(
  File "/home/arijitdasgupta/PhysTwin/benchmarks/spring_mass_gpu/topology.py", line 36, in benchmark_one_case
    trainer = load_trainer_for_case(base_path, case_name)
  File "/home/arijitdasgupta/PhysTwin/benchmarks/spring_mass_gpu/load_case.py", line 21, in load_trainer_for_case
    return load_trainer_and_model(base_path, case_name)
  File "/home/arijitdasgupta/PhysTwin/rerun_viz/replay_core.py", line 71, in load_trainer_and_model
    trainer = InvPhyTrainerWarp(
  File "/home/arijitdasgupta/PhysTwin/qqtt/engine/trainer_warp.py", line 140, in __init__
    self.simulator = SpringMassSystemWarp(
  File "/home/arijitdasgupta/PhysTwin/qqtt/model/diff_simulator/spring_mass_warp.py", line 810, in __init__
    self.step()
  File "/home/arijitdasgupta/PhysTwin/qqtt/model/diff_simulator/spring_mass_warp.py", line 1011, in step
    wp.launch(
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/context.py", line 6847, in launch
    module_exec = kernel.module.load(device, block_dim)
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/context.py", line 2925, in load
    raise (e)
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/context.py", line 2922, in load
    self._compile(device, module_dir, output_name, output_arch)
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/context.py", line 2675, in _compile
    builder = ModuleBuilder(
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/context.py", line 2053, in __init__
    self.build_kernel(kernel)
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/context.py", line 2085, in build_kernel
    kernel.adj.build(self)
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/codegen.py", line 971, in wrapper
    return func(*args, **kwargs)
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/codegen.py", line 1195, in build
    raise type(original_exc)(*new_args).with_traceback(original_exc.__traceback__) from None
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/codegen.py", line 1183, in build
    adj.eval(adj.tree.body[0])
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/codegen.py", line 3489, in eval
    return emit_node(adj, node)
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/codegen.py", line 2034, in emit_FunctionDef
    adj.eval(f)
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/codegen.py", line 3489, in eval
    return emit_node(adj, node)
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/codegen.py", line 2603, in emit_For
    adj.materialize_redefinitions(adj.loop_symbols[-1])
  File "/opt/conda/envs/phystwin/lib/python3.10/site-packages/warp/_src/codegen.py", line 2403, in materialize_redefinitions
    raise WarpCodegenError(
warp._src.codegen.WarpCodegenError: Error while parsing function "compute_neigh_indices" at /home/arijitdasgupta/PhysTwin/qqtt/model/diff_simulator/spring_mass_warp.py:403:
            min_index = j
;Error mutating a constant min_dist inside a dynamic loop, use the following syntax: pi = float(3.141) to declare a dynamic variable

```
## Methodology

- **Forward only:** one **outer** physics update per timed interval (see glossary). When `cfg.use_graph` is True, the outer step is a **CUDA graph replay** of the same work as `SpringMassSystemWarp.step()` (`wp.capture_launch(forward_graph)`). When `False`, the outer step is a direct `step()` call.
- **Outer step vs substep:** A **substep** is one index `i` in the inner loop `for i in range(num_substeps)` inside `step()`. One **outer step** is the **entire** call to `step()` — including **all** substeps (and graph capture wraps that full inner loop). Table column **“Outer step mean (ms)”** is one mean wall time for that full outer step.
- **Derived column:** **outer ms per substep** = `outer_step_mean_ms / num_substeps` (average duration per inner substep if work were uniform; no separate substep timer).
- **Collision:** production-like — loaded from case + `cfg` (see per-row flags).
- **Data root:** `./data/different_types`
- **Warmup / repeats:** 5 / 20
- **Primary benchmark path `use_graph`:** **True** (same as `cfg.use_graph` during the run).
- **Segment timing (optional):** CUDA events inside `step()` with **`cfg.use_graph=False`** for that measurement. Segment sums are **eager-only** and are **not** equal to graph-mode wall time; use them to see where work goes inside the eager pipeline.
- **Track (A) load ceiling:** `double_lift_cloth_1` — load independent `InvPhyTrainerWarp` instances until first load-time OOM (`oom_at_n = *(pending — load phase not finished or report written mid-load)*`). This is **not** the same as how many simulators can be **stepped** in one timing sweep.
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

## Track (A) — parallel instances (same case)

- **Load status:** not started yet (topology track just finished; parallel loads pending).
- **Timing summary:** **0** timing row(s); **last timed N** = **None** (largest `n` in the table below). `timing_complete` = **False**.
_Parallel timing not started yet (track B only, load in progress, or report written before the N sweep)._ 

## Model size glossary

- **Topology:** `n_vertices`, `num_object_points`, `n_springs`, control points, substeps.
- **Outer step mean (ms):** one full `step()` call (all substeps); graph path when `cfg.use_graph`.
- **spring_Y MB:** approximate bytes for stiffness log tensor.
- **Peak alloc MB:** `torch.cuda.max_memory_allocated()` around the timed region.

