# Rerun visualization (spring–mass replay)

**Default: write a `.rrd` file** — no network. Copy to your laptop and open with the Rerun app.

---

## Single case → `.rrd`

```bash
python -m rerun_viz.replay_recorded --case_name double_lift_cloth_3 [--max_frames 50] [--output-rrd my_replay.rrd]
```

Writes `replay_<case_name>.rrd` unless `--output-rrd` is set.

**Options:** `--minimal`, `--medium`, `--no-global-normalization`, `--rerun_mode file|serve|connect`, etc. See `python -m rerun_viz.replay_recorded --help`.

**CLI UX:** Warp’s device banner and qqtt `[DATA]` / `[SIMULATION]` INFO lines are suppressed on the TTY; after load you get a short **PhysTwin ready** summary (checkpoint, mesh, frames). The global color **pre-pass** shows a **`pre-pass` tqdm** progress bar. `torch.load` `FutureWarning` is filtered for that call.

**Speed (pre-pass):** The pre-pass runs full physics once per frame (like replay) to estimate global color ranges. **Faster options:** (1) **`--no-global-normalization`** — skips pre-pass entirely (per-frame colors). (2) **`--max_frames N`** — fewer frames for both pre-pass and replay. (3) **On-disk cache (default on)** — second run with the same case, frame range, and checkpoint reuses `experiments/<case>/train/.replay_prepass_<start>_<end>.pkl`. Use **`--prepass-refresh`** to recompute, **`--no-prepass-cache`** to disable cache read/write.

---

## All cases → one folder

```bash
./src/rerun_viz/replay_all_cases.sh
# or: nohup ./src/rerun_viz/replay_all_cases.sh &
```

Outputs under `./replay_rrds/`; log in `replay_all.log`. See script header for `LOG_FILE`, `BASE_PATH`.

---

## Optional: **serve** (live VM → laptop)

1. **Laptop:** SSH with local forward, e.g. `ssh -L 9876:127.0.0.1:9876 user@vm`
2. **VM:** `python -m rerun_viz.replay_recorded --case_name YOUR_CASE --rerun_mode serve`
3. **Laptop:** `rerun rerun+http://127.0.0.1:9876/proxy` while the VM process is running

Printed hints: `REMOTE_LIVE_SERVE_SSH` from `connect_instructions.py`.

---

## Optional: **connect** (Python pushes to viewer; SSH `-R`)

Viewer on laptop first, then:

```bash
ssh -R 9876:127.0.0.1:9876 user@your-vm
python -m rerun_viz.replay_recorded ... --rerun_mode connect
```

May show flush timeouts over SSH; **serve** is usually easier. See `JTAP_STYLE_SSH` in `connect_instructions.py`.

---

## Other modules

| Module | Role |
|--------|------|
| `replay_core` | Replay loop + Rerun streaming (imported by `replay_recorded`) |
| `case_setup` | Shared `cfg` loading (yaml + optimal params + camera) |
| `terminal_output` | TTY-colored status lines for replay |
| `spring_mass_logging` | Logging helpers for replay |
| `inspect_case_stats` | Print mass/stiffness stats (same asserts as replay) |
| `port_util` | Free-port helpers for `serve` |

---

## Tests

```bash
pytest tests/test_spring_mass_logging.py tests/test_port_util.py tests/test_case_setup.py \
  tests/test_terminal_output.py tests/test_replay_helpers.py -q
```

Format / lint (requires [Ruff](https://docs.astral.sh/ruff/)):

```bash
ruff check src/rerun_viz tests
ruff format src/rerun_viz tests
```

Match **Rerun app** version to **rerun_sdk** (e.g. 0.30.x).
