# Plan: rerun_viz quality pass (minions)

**Status:** DONE (Implementer + user answers incorporated).

**Goal:** Remove redundancy, clarify structure, apply consistent formatting, and extend tests so `rerun_viz/` stays correct and maintainable without changing intended behavior (replay, colors, serve/connect, docs).

---

## Scope (in)

- `rerun_viz/__init__.py`, `README.md`, `connect_instructions.py`
- `spring_mass_logging.py` (pure helpers + Rerun logging)
- `replay_recorded.py` (CLI + replay loop + global color pre-pass)
- `port_util.py`
- `inspect_case_stats.py`
- `minimal_rerun_serve_test.py` (purpose: manual/quick serve check — clarify or fold into docs)
- `replay_all_cases.sh`
- Tests: `tests/test_spring_mass_logging.py`, `tests/test_port_util.py`; add new tests where gaps exist

## Scope (out unless user opts in)

- Gradio / `interactive_playground_gradio.py`
- Training / `trainer_warp.py` behavior changes (only if required to fix a broken import after refactor)
- CI wiring (GitHub Actions) unless user asks

---

## Observations (baseline)

- **`replay_recorded.py`** is large (~500+ lines): config loading, trainer load, global color collection, main replay loop, argparse. Overlap with **`inspect_case_stats.py`** (yaml case type, `optimal_params`, `InvPhyTrainerWarp` construction, `best_*.pth` glob).
- **`spring_mass_logging.py`**: module-level globals for one-shot legends; `import rerun` after globals — cosmetic reorder for clarity.
- **`port_util.py`**: small, tested; keep as single source for serve port logic.
- **Tests:** Good coverage for strips, colors, smoke Rerun logs; **no** automated test for `replay_recorded` or `inspect_case_stats` (heavy deps / data). Mitigation: extract **pure** helpers (e.g. shared “case config” dict or small functions) and test those; optional **skipped** integration test behind env var.
- **Formatting:** No root `pyproject.toml` / pre-commit found — use consistent 4-space indent, docstrings, and optional `black`/`ruff` only if we add config in this pass (user decision).

---

## Proposed work items (checklist)

### A. Deduplicate and clarify

- [ ] **A1.** Extract shared “load case config” logic used by both `replay_recorded.load_config_and_camera` and `inspect_case_stats` into a small module under `rerun_viz/` (e.g. `case_setup.py` or `phys_case_config.py`) with:
  - `resolve_config_yaml(case_name: str) -> str` (path to yaml)
  - `load_optimal_params(case_name: str) -> object` (or typed if easy)
  - Optionally shared `load_calibrate_and_metadata(base_path, case_name)` for replay only
- [ ] **A2.** Keep **replay-specific** pieces (camera, `cfg.c2ws`, logger paths) in `replay_recorded` or import from the new module without circular imports.
- [ ] **A3.** Document in README that `inspect_case_stats` is a **diagnostic** script (requires same data/checkpoints as replay).

### B. `spring_mass_logging.py` cleanup

- [ ] **B1.** Reorder imports (stdlib → third party → local); move legend globals next to legend helpers or document why they exist.
- [ ] **B2.** Scan for dead code / duplicate color paths; ensure stretch vs stiffness colormaps stay as tested.
- [ ] **B3.** Short module docstring at top describing layers: pure numpy → torch/warp → rr logging.

### C. `replay_recorded.py` structure

- [ ] **C1.** Either split into `replay_recorded.py` (CLI only) + `replay_core.py` (functions), **or** keep one file but add clear section comments — **user preference** (see questions).
- [ ] **C2.** Ensure `collect_global_color_ranges` and main loop share one definition of “object-object spring” filtering (single helper if duplicated).
- [ ] **C3.** `set_all_seeds`: confirm single call site; document why seeds matter for replay.

### D. Formatting and packaging

- [ ] **D1.** Apply consistent formatting to all touched files (PEP8; run formatter only if user approves adding minimal config).
- [ ] **D2.** `README.md`: table of scripts, env assumptions, test command (`pytest tests/test_spring_mass_logging.py tests/test_port_util.py`).

### E. Tests

- [ ] **E1.** Add unit tests for any **new pure helpers** (e.g. yaml resolution, optimal path string).
- [ ] **E2.** Add tests for `compute_global_color_ranges` / edge cases if not already covered (empty stretch list, single frame).
- [ ] **E3.** Run full relevant pytest subset and record command + result in this plan’s footer when done.

### F. Verifier loop

- [ ] **F1.** Verifier runs pytest + quick grep for duplicate large blocks.
- [ ] **F2.** Implementer addresses failures until Verifier **PASS**.

---

## Acceptance criteria

1. No intentional behavior change for: default replay path, global normalization defaults, serve/connect messaging, spring/mass/collision logging semantics.
2. `pytest tests/test_spring_mass_logging.py tests/test_port_util.py` passes (and any new test files added).
3. Redundant config/trainer setup code between `replay_recorded` and `inspect_case_stats` is reduced via shared module or documented why not.
4. README lists entry points and how to run tests.
5. Code is readable: sections or modules, no obvious dead code.

---

## Risks

- **Refactor breakage:** Shared module must not create import cycles with `qqtt` or `rerun_viz.spring_mass_logging`.
- **Integration untested:** Full replay still manual; mitigate with pure-function tests and smoke tests.

---

## Planner questions (answer with numbers)

1. **File split:** Do you want `replay_recorded.py` **split** into e.g. `replay_core.py` + thin CLI, or **keep one file** with section headers only?
2. **Tooling:** OK to add a minimal **`pyproject.toml`** (or `setup.cfg`) with **black**/`ruff`** for `rerun_viz/` and `tests/`, or prefer **manual** PEP8 only?
3. **`minimal_rerun_serve_test.py`:** Keep as standalone script, move examples to README only, or **delete** if redundant with `replay_recorded --rerun_mode serve`?
4. **Integration test:** Add **`pytest.mark.skip`** integration test that loads a tiny fixture (or env `PHYS_TWIN_REPLAY_SMOKE=1` + path to case), or **skip** and rely on unit tests only?
5. **Strictness:** Should `inspect_case_stats` use the **same** assertions as replay for missing `optimal_params.pkl` (fail fast) vs current softer “print and return” when no checkpoint?

---

## Footer (Implementer fills)

- **Planner sign-off:** user answered (split + ruff + remove minimal test + more tests + match replay + clean terminal)
- **Verifier result:** PASS — `pytest tests/test_spring_mass_logging.py tests/test_port_util.py tests/test_case_setup.py tests/test_terminal_output.py tests/test_replay_helpers.py` → **30 passed**; `ruff check rerun_viz tests` → clean; `python -m rerun_viz.replay_recorded --help` OK
- **pytest command & result:** see above
