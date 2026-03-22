# Optional dependencies

Heavy or platform-specific stacks are **not** required for the default CPU test subset.

| Extra / area | Notes |
|--------------|--------|
| `pyproject` `[project.optional-dependencies]` | `dev` (pytest, ruff), `viz`, `gradio`, `gaussian`, etc. |
| **PyTorch CUDA** | Install from the PyTorch index URL matching your driver; see `docs/UV_AND_PIXI.md`. |
| **pytorch3d** | Often installed from a Facebook-hosted wheel URL; see legacy `env_install/env_install.sh`. |
| **GroundingDINO / TRELLIS / RealSense** | Documented in `env_install/` scripts; manual steps. |
| **kornia / gsplat** | Declared as **core** dependencies in `pyproject.toml` (the trainer imports them at load time). |
| **einops / trimesh / torchvision** | **Core** deps in `pyproject.toml` (Gaussian / scene code paths). |
| **pytorch3d** | Needed for **`scripts/entrypoints/gaussian/gs_render.py`** (ball-query / mesh distance) and **full** Gaussian render CLIs — **not** required for `python -m rerun_viz.replay_recorded` (trainer uses `gaussian_splatting/prune_utils.py` instead of importing `gs_render`). |
| **Gaussian submodules** (`diff_gaussian_rasterization`, `simple-knn`, …) | **Lazy-imported** where possible: `diff_gaussian_rasterization` only loads for `use_gsplat=False` rendering; `simple_knn` only when initializing Gaussians from PCD. Replay / training with gsplat path can run without building them in many setups. |
