# `qqtt` package

Core PhysTwin library: **spring–mass simulation** (Nvidia Warp), **trainers**, **data** utilities, and **config** (`qqtt.utils.cfg`).

- Entry re-exports in `qqtt/__init__.py` use lazy loading (`InvPhyTrainerWarp`, `SpringMassSystemWarp`, `OptimizerCMA`) so `import qqtt` stays light until those names are accessed.
- Full training/inference still requires vendored `gaussian_splatting` on `PYTHONPATH` when the trainer imports Gaussian rendering.

See `docs/agents/architecture.md` and root `README.md`.
