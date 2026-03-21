# Rerun visualization (spring–mass replay)

**Default: write to `.rrd` file** — no network. Copy to laptop and `rerun replay_<case>.rrd`.

---

## Default: **file** (write .rrd)

```bash
python -m rerun_viz.replay_recorded --case_name double_lift_cloth_3 [--max_frames 50] [--output-rrd my_replay.rrd]
```

Writes `replay_<case_name>.rrd` (or `--output-rrd`). Copy to your laptop and open with `rerun replay_*.rrd`.

---

## Optional: **serve** (live VM → laptop)

1. **Laptop:** open SSH with a **local** forward (traffic: laptop `127.0.0.1:9876` → VM `127.0.0.1:9876`):

   ```bash
   ssh -L 9876:127.0.0.1:9876 user@your-vm
   ```

   Example `~/.ssh/config` (use **either** `LocalForward` **or** `RemoteForward` for port 9876 — not both):

   ```
   Host my-vm
     HostName ...
     User ...
     LocalForward 9876 127.0.0.1:9876
   ```

2. **VM:** run with `--rerun_mode serve`:

   ```bash
   python -m rerun_viz.jtap_style_demo
   python -m rerun_viz.demo_random_points --seconds 30
   python -m rerun_viz.replay_recorded --case_name YOUR_CASE ...
   ```

3. **Laptop** (separate terminal, while SSH stays connected), **while the VM script is still running**:

   ```bash
   rerun rerun+http://127.0.0.1:9876/proxy
   ```

   If the Python process on the VM has already exited, **nothing is listening** — connect before stopping the script, or use `jtap_style_demo` which **keeps the server alive** until you press Ctrl+C on the VM.

Match **Rerun app** version to **rerun_sdk** (e.g. both 0.30.x).

---

## Alternative: **connect** (Python pushes to viewer; SSH `-R`)

Viewer on laptop first, then:

```bash
ssh -R 9876:127.0.0.1:9876 user@your-vm
```

Then on VM:

```bash
python -m rerun_viz.replay_recorded ... --rerun_mode connect
```

This often shows **flush timeouts** over SSH; prefer **serve** above.

---

## File-only (no network)

```bash
python -m rerun_viz.jtap_style_demo --mode file --rrd-path /tmp/demo.rrd
```

Copy `demo.rrd` to the laptop and: `rerun demo.rrd`

---

## Helpers

- `minimal_rerun_serve_test` — tiny serve-mode check.
- `connect_instructions.py` — text blocks printed by the scripts (`REMOTE_LIVE_SERVE_SSH`, `JTAP_STYLE_SSH`).
