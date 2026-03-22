# PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos

**Authors:** Hanxiao Jiang, Hao-Yu Hsu, Kaifeng Zhang, Hsin-Ni Yu, Shenlong Wang, Yunzhu Li  
**Affiliations:** Columbia University; University of Illinois Urbana-Champaign

**Links:** [Website](https://jianghanxiao.github.io/phystwin-web/) · [Paper](https://jianghanxiao.github.io/phystwin-web/phystwin.pdf) · [arXiv](https://arxiv.org/abs/2503.17973)

![Teaser](./assets/teaser.png)

This repository contains the official implementation of **PhysTwin**. The core Python packages live under **`src/`** (`qqtt`, `rerun_viz`); benchmarks ship as the **`benchmarks`** package. **CLI-style scripts** live under **`scripts/shims/`** (thin files that delegate to `scripts/entrypoints/...`); run them from the repo root, e.g. `python scripts/shims/train_warp.py` (see `docs/agents/repo-layout.md`).

---

## Install (UV, recommended)

**Prerequisites:** Linux with NVIDIA GPU for full training/inference; Python 3.10+; CUDA toolchain compatible with your PyTorch build.

1. **Install [uv](https://docs.astral.sh/uv/).**

2. **Clone and sync** from the repository root:

   ```bash
   cd PhysTwin
   chmod +x env_install/install.sh
   ./env_install/install.sh
   ```

   This creates `.venv` and installs the project in editable mode (`phystwin` on `pyproject.toml`).

3. **PyTorch with CUDA** (pick the index URL that matches your driver; see [PyTorch Get Started](https://pytorch.org/get-started/locally/)):

   ```bash
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Vendored Gaussian extensions** (required for trainer paths that import `gaussian_splatting`):

   ```bash
   export PYTHONPATH="$(pwd)/scripts/shims:$(pwd):${PYTHONPATH}"
   pip install --no-build-isolation gaussian_splatting/submodules/diff-gaussian-rasterization/
   pip install --no-build-isolation gaussian_splatting/submodules/simple-knn/
   ```

5. **Verify imports:**

   ```bash
   uv run python -c "import qqtt, rerun_viz"
   ```

   Full `import qqtt` after extensions are built: see `docs/agents/testing.md`.

**Lockfile:** `uv.lock` is committed for reproducible installs (`uv sync --frozen`).

**Optional:** [Pixi](https://pixi.sh/) notes — `docs/UV_AND_PIXI.md`. Legacy **conda** scripts under `env_install/` are **deprecated** for new setups but retained for reference.

**Docker (UV-based):**

```bash
docker build -t phystwin:uv .
docker run --gpus all -it --rm -v "$PWD":/work -w /work phystwin:uv bash
```

---

## Development

```bash
uv run ruff check src tests benchmarks scripts
uv run ruff format --check src tests benchmarks scripts
uv run pytest -q -m "not gpu"    # CPU CI subset; omit -m to include GPU-marked tests locally
```

See `docs/agents/testing.md` and `.github/workflows/ci.yml`.

---

## Run (key commands)

Paths like `experiments/`, `data/`, and `experiments_optimization/` are unchanged relative to the repo root. Implementation files for the following live under **`scripts/entrypoints/`** (by category: `train/`, `inference/`, `optimize/`, `data/`, `gaussian/`, `eval/`, `playground/`); batch inputs such as `data_config.csv` are under **`config/`**.

| Task | Command |
|------|---------|
| Interactive playground | `python scripts/shims/interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_sloth` |
| Zero-order optimization | `python scripts/shims/script_optimize.py` |
| First-order training | `python scripts/shims/script_train.py` |
| Inference | `python scripts/shims/script_inference.py` |
| Gaussian (first frame) | `bash scripts/shell/gs_run.sh` |
| Rerun replay → `.rrd` | `python -m rerun_viz.replay_recorded --case_name double_lift_cloth_3` |
| Spring–mass GPU benchmark | `python scripts/bench_spring_mass_gpu.py --output_dir benchmarks/reports/run_001` |

**Gradio:** `python scripts/shims/interactive_playground_gradio.py`, `python scripts/shims/run_playground_gradio.py`, or `bash scripts/shell/run_gradio_filtered.sh`.

**Data & checkpoints (downloads):**

- [data.zip](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip)
- [experiments_optimization.zip](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments_optimization.zip)
- [experiments.zip](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments.zip)
- [gaussian_output.zip](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/gaussian_output.zip)
- [(optional) additional_data.zip](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/additional_data.zip)

**Pretrained models for data processing:** `bash env_install/download_pretrained_models.sh` (when using those pipelines).

**RTX 5090 / CUDA 12.8:** see comments in `env_install/5090_env_install.sh` (legacy conda path).

**Windows:** community setup exists on branch `windows_setup`.

### Migration (paths)

If you used **`python train_warp.py`** or **`bash gs_run.sh` from the repo root**, those files were removed to keep the root clean. Use **`python scripts/shims/train_warp.py`** and **`bash scripts/shell/gs_run.sh`** instead. Set `PYTHONPATH="$(pwd)/scripts/shims:$(pwd):…"` when building extensions or importing `gs_render` by module name (see `docs/agents/imports-and-packages.md`).

---

## Documentation

- **Spring–mass & data structures:** `docs/SPRING_MASS_AND_DATA_STRUCTURES.md` (module paths refer to packages under `src/qqtt/`).
- **Agents / codebase map:** `AGENTS.md`, `docs/agents/` (including **`docs/agents/repo-layout.md`** for root vs `scripts/entrypoints/`).

---

## Projects using PhysTwin

- [NovaFlow](https://novaflow.lhy.xyz/)
- [Real2Sim-Eval](https://real2sim-eval.github.io/)
- [PhysWorld](https://arxiv.org/abs/2510.21447)
- [NeuSpring](https://arxiv.org/abs/2511.08310)

---

## Citation

```bibtex
@article{jiang2025phystwin,
  title={PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos},
  author={Jiang, Hanxiao and Hsu, Hao-Yu and Zhang, Kaifeng and Yu, Hsin-Ni and Wang, Shenlong and Li, Yunzhu},
  journal={ICCV},
  year={2025}
}
```

---

## Acknowledgments

This project builds on ideas and code from the PhysTwin authors and contributors. Upstream reference repository (placeholder until canonical URL is finalized): [https://github.com/Jianghanxiao/PhysTwin](https://github.com/Jianghanxiao/PhysTwin)