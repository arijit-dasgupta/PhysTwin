# PhysTwin — repo restructure, UV tooling, and agent-oriented docs

**Status:** Phase G done — **minimal repo root** (no top-level `*.py` / `*.sh`); CLI shims under `scripts/shims/`, bash under `scripts/shell/`  
**Goal:** Major **structural and tooling** refactor with **zero functional change**: same behavior for training, inference, benchmarks, `rerun_viz`, Gradio, shell entrypoints (names preserved where listed below), and documented migration only if a name must change.  

**User feedback (2026-03-22):** Despite `src/` + packaging improvements, the **repository root still has far too many files** and looks messy. **Organization must cover the full repo**, not only `src/` and `tests/`: scripts, shell wrappers, loose markdown/data specs, eval/export utilities, etc. should live under **clear top-level buckets** (e.g. `scripts/` subtrees, `docs/`, `config/`) so the root is **mostly** `pyproject.toml`, `README.md`, `AGENTS.md`, `uv.lock`, `Dockerfile`, and thin **compatibility shims** where needed.  
**User feedback (2026-03-10):** Phase F still left **22 root `*.py` shims** — unacceptable; **move all** to `scripts/shims/`, remove root shell wrappers, document `PYTHONPATH` + new commands (**Phase G**).  
**Non-goals (explicit):**
- Rewriting physics, Warp kernels, or training algorithms.
- Changing `gaussian_splatting/submodules/*` vendored code beyond import/path fixes required by layout.
- Semantic changes to experiment outputs, metrics, or default hyperparameters.
- Replacing PyTorch / CUDA stack with a different major version (pin compatibility in `pyproject`; match current working pins from `env_install/` unless a security bump is required).

---

## Related existing plans (do not duplicate work)

| Plan | Relationship |
|------|----------------|
| [`.cursor/plans/spring-mass-benchmark-gpu-scaling.md`](spring-mass-benchmark-gpu-scaling.md) | Benchmarks already evolving; restructure must keep `benchmarks/` + tests importable. |
| [`.cursor/plans/spring-mass-performance-benchmark.md`](spring-mass-performance-benchmark.md) | Same. |
| [`.cursor/plans/spring-mass-benchmark-report-depth.md`](spring-mass-benchmark-report-depth.md) | Same. |
| [`.cursor/plans/rerun-viz-quality-pass.md`](rerun-viz-quality-pass.md), [`.cursor/plans/rerun-viz-cleanup.md`](rerun-viz-cleanup.md) | `rerun_viz/` behavior; preserve CLI and `sys.path` behavior or replace with editable install. |

---

## Current inventory (top level — brief)

**Packages / app code**
- `qqtt/` — core library (engine, models, utils); primary import root `import qqtt...`.
- `rerun_viz/` — Rerun visualization helpers, replay scripts (`replay_core.py`, `replay_recorded.py`, etc.); uses `sys.path` injection for repo root.
- `benchmarks/` — spring-mass GPU benchmarks (`benchmarks/spring_mass_gpu/`), reports under `benchmarks/reports/`.
- `scripts/` — e.g. `bench_spring_mass_gpu.py` (sys.path pattern).
- `gaussian_splatting/` — third-party-style tree with `submodules/` (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`) with local `setup.py` installs.
- `data_process/` — pipelines; may clone `TRELLIS` (see `env_install/env_install.sh`).
- `configs/`, `docs/`, `assets/`.

**Entry surfaces (Phase G):** thin **`scripts/shims/*.py`** (same names as before: `train_warp.py`, `gs_render.py`, …); bash **`scripts/shell/*.sh`**. No duplicate wrappers at repo root.

**Environment / tooling today**
- `pyproject.toml` — **Ruff-only** (no `[project]` / install metadata).
- `env_install/env_install.sh`, `env_install/5090_env_install.sh` — **conda** + `pip` for PyTorch, Warp, pytorch3d wheels, optional heavy deps.
- `Dockerfile` — Miniconda + `env_install.sh`.
- `.github/` — **workflows/ci.yml** (lint + CPU pytest) plus any other org defaults (e.g. FUNDING).

**Tests**
- `tests/` — pytest modules; several prepend `PROJECT_ROOT` to `sys.path` (same pattern as `rerun_viz`).

**Large / generated artifacts in repo (do not move blindly)**
- `experiments/`, `experiments_optimization/`, `gaussian_output/`, `temp_experiments/`, `replay_rrds/`, `.rrd` files — treat as data/output; restructuring should avoid breaking documented paths in README/scripts.

---

## Design decision: package layout (`src/` vs flat `qqtt/`)

**Recommendation:** Adopt a **`src/` layout** with **unchanged Python package names** `qqtt` and `rerun_viz` (i.e. `src/qqtt/`, `src/rerun_viz/`).

**Justification**
- **Minimal import churn:** `import qqtt` and `import rerun_viz` stay valid after `pip install -e .` (or uv equivalent) without renaming hundreds of modules.
- **Clear separation:** Vendored / standalone trees (`gaussian_splatting/`, `data_process/` clones) remain **outside** `src/` until a later initiative explicitly namespaces them.
- **Alternative considered:** Rename to `src/phystwin/` and nest old code under `phystwin/qqtt` — **high blast radius** and violates “zero functional change” unless exhaustive shims + deprecation; **defer**.

**Root cleanup strategy**
- Move **non-entry** modules into `scripts/` or `tools/` only when paired with **thin root wrappers** or **console_scripts** so documented commands keep working OR README lists the new invocation (prefer preserving names).
- Group loose markdown (`downsampling_spring_mass.md`) under `docs/` with **redirect note in old path** only if anything references the old filename (grep); otherwise move without symlink (avoid clutter).

---

## Import path strategy

1. **Primary:** Declare packages in `pyproject.toml` (`[build-system]` + `hatchling` or `setuptools`, `packages` under `src/`). Developer workflow: **`uv sync`** / **`uv pip install -e .`** so `qqtt` / `rerun_viz` import without `PYTHONPATH`.
2. **Remove `sys.path` hacks where obsolete:** After editable install works in CI and local dev, replace inserts in `rerun_viz/*.py`, `tests/`, `benchmarks/`, `scripts/` with normal imports; keep **one** compatibility path if a script must run as `python path/to/script.py` from an unpacked tarball without install — document as secondary.
3. **Shim modules (only if needed):** If any external docs reference a **moved file path**, add a **deprecated shim** file at the old location that re-exports or prints a one-time `warnings.warn` — **time-boxed** (e.g. one release) and listed in README “Migration”.

---

## Dependency management: UV first, Pixi fallback

**Target:** Remove **conda** from documented workflows and from `Dockerfile`; use **UV** for environment + lockfile.

**Phase C — steps (checklist)**
- [x] Encode **runtime** and **optional** dependencies in `pyproject.toml` (`[project]` / `[project.optional-dependencies]`): e.g. `dev` (pytest, ruff), `viz` (rerun), `gradio`, `bench-gpu` (warp, numpy stack), `gaussian` (gsplat, kornia, local path deps).
- [ ] **Pin strategy:** Start from `env_install/env_install.sh` versions; use **UV’s** resolver; generate **`uv.lock`** committed to repo.
- [x] **PyTorch + CUDA:** Use UV’s documented approach (e.g. extra index URL for PyTorch wheels) — document **one** blessed command block in README for Linux + CUDA 12.x; note CPU-only as non-default.
- [x] **Heavy / awkward packages** (pytorch3d, GroundingDINO, TRELLIS): keep as **optional** extras or **documented manual** steps; do not block `uv sync --extra dev` for core tests.
- [x] **Pixi fallback:** If a system library (GL, Qt, etc.) blocks UV on a given machine, document **`pixi.toml`** as **optional** reproducible env with same version pins; clarify **when** to use (e.g. headless vs GUI).
- [ ] **Remove or archive conda:** Delete conda install lines from `env_install/*.sh` **or** replace scripts with `uv`-based installers; update `Dockerfile` to **UV-based** multi-stage image (no Miniconda).
- [x] **Verification:** Fresh clone → `uv sync` → `pytest` (CPU-safe subset) passes; optional job with GPU extras where available.

---

## Phased rollout + verifier gates

| Phase | Content | Verifier gate (exit criteria) |
|-------|---------|-------------------------------|
| **A** | Create `src/qqtt`, `src/rerun_viz`; fix packaging metadata; adjust imports / remove redundant `sys.path` where safe; thin root wrappers if needed. | Import smoke: `python -c "import qqtt, rerun_viz"`; key CLIs launch to `--help` without error. |
| **B** | Tests: ensure `pytest` from repo root; add tests for packaging (import paths), optional markers for GPU-heavy tests; **no behavior change**. | `pytest` green on CPU CI subset; GPU tests skipped or run in optional workflow. |
| **C** | UV: full `pyproject`, `uv.lock`, Dockerfile, delete conda-centric paths; optional pixi doc. | Lockfile reproducible install; Dockerfile builds. |
| **D** | README **full rewrite** (last): install, dev, run key commands; **bottom:** thank original authors + link placeholder `https://github.com/.../PhysTwin` until user supplies URL. | Human-readable; links work; commands copy-paste. |
| **E** | Agent-oriented docs + Cursor rules/skills (see below). | Rules reference docs; docs list owners. |
| **F** | **Full-repo taxonomy:** implementations under `scripts/entrypoints/`, `config/`, docs moves. | `pytest -m "not gpu"` green. |
| **G** | **Minimal root:** move **all** thin `*.py` from root → **`scripts/shims/`**; delete root **`*.sh`** one-liners; README + `PYTHONPATH` + shell sources; subprocess strings in `script_*.py`. | **Zero** top-level `*.py` / `*.sh` entry scripts; `pytest -m "not gpu"` green; `python scripts/shims/train_warp.py --help` works from repo root. |

Between phases: **no physics/output changes** — diff should be structure, imports, tooling, docs.

---

## Agent-oriented documentation & Cursor integration

**Deliverables**
- [x] Root **`AGENTS.md`** — map: areas of codebase (`qqtt/engine`, `rerun_viz`, `benchmarks`, `gaussian_splatting`) + “read before edit” pointers.
- [x] **`docs/agents/`** — high-density topic files, e.g. `architecture.md`, `imports-and-packages.md`, `optional-dependencies.md`, `testing.md`.
- [x] **Co-located READMEs** where high churn: `benchmarks/README.md`, `rerun_viz/README.md` (extend existing), `src/qqtt/README.md` (short).
- [x] **Cursor rules** (`.cursor/rules/`): e.g. `phystwin-codebase.mdc` — (a) when editing a subtree, **open** the linked agent doc; (b) if change affects behavior or public API surface, **update** agent markdown in same PR.
- [ ] **Optional skill** (`.cursor/skills/` or user skills): “update agent docs when changing imports/entrypoints” — only if repo convention warrants; keep minimal.

---

## Tests & CI

**Current:** **`.github/workflows/ci.yml`** runs `uv sync --extra dev`, `ruff check` / `ruff format --check` on `src`, `tests`, `benchmarks`, `scripts`, and `pytest -m "not gpu"`.  
**Add tests where valuable:** packaging/import tests, one test that `python -m rerun_viz` or equivalent entrypoint resolves (if added), optional snapshot of `--help` for main scripts.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Circular imports after `src/` move | Incremental moves; run `python -m compileall`; use lazy imports only if already pattern in codebase. |
| Warp/CUDA compile/runtime | Do not change Warp usage; CI skips GPU; document driver/CUDA alignment with torch index. |
| Optional deps (open3d, gradio, rerun) | Extras in `pyproject`; graceful skip in tests via `pytest.importorskip`. |
| pytorch3d / custom wheels | Keep manual wheel URL in docs; optional extra. |
| Dockerfile size / build time | Multi-stage; cache UV layers; align with README. |
| Path assumptions in scripts | Grep for hardcoded `qqtt/` paths; fix to use `importlib.resources` or repo-root detection only where unavoidable. |

---

## File inventory (expected touch areas)

- **New / major:** `pyproject.toml` (full project metadata), `uv.lock`, `src/qqtt/**`, `src/rerun_viz/**`, `.github/workflows/ci.yml`, `AGENTS.md`, `docs/agents/*.md`, `.cursor/rules/*.mdc`, rewritten `README.md`, `Dockerfile`, `env_install/*` (UV-based).
- **Update:** All modules that use `sys.path` injection; `benchmarks/`, `tests/`, `scripts/`; `.cursor/plans/README.md` cross-link optional.
- **Preserve behavior:** `qqtt` public scripts API if any external code depends on file paths (grep before deleting root files).

---

## Acceptance criteria (Definition of Done)

- [ ] **Functional parity:** Training/inference/benchmark/rerun/gradio entrypoints behave as before (same flags where applicable; same outputs on fixed seeds/fixtures).
- [x] **No conda** in primary developer path; **UV** documented; **Pixi** documented as fallback.
- [x] **Installable package:** `pip install -e .` / `uv sync` yields working `import qqtt`, `import rerun_viz`.
- [x] **README** complete rewrite with install/dev/commands; **acknowledgments + original repo link** at bottom (placeholder OK).
- [x] **Agent docs + Cursor rules** in place; rules require reading/updating docs when appropriate.
- [x] **Full-repo taxonomy (Phase F + G):** implementations under `scripts/entrypoints/`; **CLI shims** under **`scripts/shims/`** (not repo root); **`pytest` `pythonpath`** + Docker **`PYTHONPATH`** include `scripts/shims` for `from gs_render import ...`.
- [x] **CI** runs lint + pytest (CPU); GPU tests optional or manual.
- [x] **Migration notes** in README (paths: `scripts/shims/`, `scripts/shell/`).

### Verifier gate (2026-03-22)

With dev dependencies available (`uv sync --extra dev` or equivalent), from the repo root the following exit **0** (matches `.github/workflows/ci.yml` scope for Ruff):

- `ruff check src tests benchmarks scripts`
- `ruff format --check src tests benchmarks scripts`
- `pytest -m "not gpu"`

*(Legacy bodies live under `scripts/entrypoints/` and are excluded from Ruff in `pyproject.toml`; `scripts/shims/*.py`, `_compat_shim.py`, and `scripts/bench_spring_mass_gpu.py` are in CI scope.)*

**Phase F + G:** `docs/agents/repo-layout.md`; `scripts/shims/` + `scripts/shell/`; `config/`; `pytest` `pythonpath` includes `scripts/shims`.

---

## Open questions / defaults applied

1. **Original PhysTwin GitHub URL:** Use placeholder `https://github.com/.../PhysTwin` in README until the user provides the canonical upstream URL (Implementer fills on user input).
2. **CI runner:** Default **GitHub Actions `ubuntu-latest`** for lint + CPU pytest; no GPU in default CI unless requested later.
3. **Package rename:** Default **keep `qqtt` and `rerun_viz` names** under `src/` to minimize risk.

---

## Suggested first phase for Implementer

**Phase A:** Introduce `src/` layout + `pyproject.toml` package tables; get editable install working; mechanically adjust imports and remove redundant `sys.path` blocks; keep root script names stable via wrappers or unchanged paths until Phase D README consolidation.

**Phase F (done):** Implementations under `scripts/entrypoints/`; initial root shims (later removed in G).  
**Phase G (done):** Root shims → `scripts/shims/`; root shell wrappers deleted; README / Dockerfile / `pyproject.toml` `pythonpath` / subprocess strings updated.
