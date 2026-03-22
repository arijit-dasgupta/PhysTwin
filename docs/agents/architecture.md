# Architecture (high level)

## Simulation

- **Spring–mass on GPU:** `SpringMassSystemWarp` in `src/qqtt/model/diff_simulator/spring_mass_warp.py` (Nvidia Warp).
- **Trainer:** `InvPhyTrainerWarp` in `src/qqtt/engine/trainer_warp.py` loads data, builds the simulator, runs optimization or inference.
- **CMA:** `OptimizerCMA` in `src/qqtt/engine/cma_optimize_warp.py`.

## Visualization

- **Rerun:** `src/rerun_viz/replay_core.py` loads a case (config, checkpoint, camera) the same way as training/inference scripts, then streams state to Rerun.
- **CLI:** `python -m rerun_viz.replay_recorded` (see `src/rerun_viz/README.md`).

## Benchmarks

- **Spring–mass GPU:** `benchmarks/spring_mass_gpu/` — discovery, topology, parallel load-to-OOM, reporting. Invoked via `scripts/bench_spring_mass_gpu.py` at repo root.

## Vendored rendering

- **`gaussian_splatting/`** is on-disk next to the repo; imports expect the repo root on `PYTHONPATH`. Extensions (`simple-knn`, `diff-gaussian-rasterization`) are built with local `pip install` from `gaussian_splatting/submodules/`.
