# rerun_viz cleanup

**Status:** done (tests: `pytest tests/test_spring_mass_logging.py tests/test_port_util.py`)  
**Goal:** Remove demo-only scripts, keep replay + helpers, tighten README and connect text.

## Removed

- `demo_random_points.py` — demo
- `jtap_style_demo.py` — demo
- `debug_stretch_colors.py` — one-off stretch debug (bug fixed)

## Kept

- `replay_recorded.py`, `replay_all_cases.sh`, `spring_mass_logging.py`
- `inspect_case_stats.py`, `port_util.py`, `connect_instructions.py`
- `minimal_rerun_serve_test.py` — no-PhysTwin serve smoke test (not a “demo” of product)

## Docs

- `README.md` — focus on file replay + serve/connect + batch script
- `connect_instructions.py` — drop jtap references
- `__init__.py` — accurate package summary
