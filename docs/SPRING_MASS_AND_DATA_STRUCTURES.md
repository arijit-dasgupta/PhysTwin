# PhysTwin: Spring-Mass Model and Data Structures Guide

This document explains the repo layout, where the **spring-mass forward step** lives, the **data structures** involved, and how **actions** (control points) are recorded in the Gradio playground and applied in the forward model.

---

## 1. Repo overview (high level)

- **PhysTwin**: Physics-informed reconstruction and simulation of deformable objects from videos (ICCV 2025).
- **Pipeline**: Raw video → processed data (`final_data.pkl`) → optimization (CMA-ES then gradient-based) → trained spring-mass params → **interactive playground** (keyboard/Gradio) or inference.
- **Core simulation**: Differentiable spring-mass system implemented in **Nvidia Warp** (GPU), in `src/qqtt/model/diff_simulator/spring_mass_warp.py`.
- **Training / inference / playground**: `src/qqtt/engine/trainer_warp.py` builds the simulator from data and runs optimization or forward rollout.

---

## 2. Where the spring-mass forward step lives

**File:** `src/qqtt/model/diff_simulator/spring_mass_warp.py`

- **Main entry point for one “logical” timestep:** `SpringMassSystemWarp.step()` (around lines 973–1032).
- One call to `step()` advances time by **one frame** by running **`num_substeps`** internal substeps (e.g. 10). So “one timestep” in the sense of one frame = one `step()` = multiple substeps.

**What `step()` does (per substep):**

1. **Control points (actions):** If `controller_points` is set, it interpolates control positions from “original” to “target” for the current substep via the Warp kernel `set_control_points`.
2. **Spring forces:** Launches `eval_springs` (positions + velocities + control positions/velocities → forces on object vertices).
3. **Velocity update from forces:** Launches `update_vel_from_force` (forces + gravity + drag → new velocities).
4. **Optional object–object collision:** If enabled, `object_collision` updates velocities.
5. **Ground collision + integration:** Launches `integrate_ground_collision` to get new positions and velocities (semi-implicit style: velocity then position update, with ground bounce).

So the **spring-mass forward function for the timestep** is effectively the full `step()` method, which composes these Warp kernels. The actual “spring law” and time integration are inside the kernels below.

---

## 3. Core kernels (math) and data they use

### 3.1 `set_control_points` (lines 84–99)

- **Role:** Defines where the “hand” control points are at the current **substep**.
- **Inputs:** `original_control_point`, `target_control_point` (both `wp.array(dtype=wp.vec3)`), substep index `step`, `num_substeps`.
- **Output:** `control_x` (control positions for this substep).
- **Formula:** Linear interpolation in time within the frame:
  - `t = (step + 1) / num_substeps`
  - `control_x[i] = original[i] + (target[i] - original[i]) * t`

So over one `step()` call, control points move smoothly from “original” to “target” across substeps.

### 3.2 `eval_springs` (lines 102–157)

- **Role:** Compute spring + dashpot forces on **object** vertices (and conceptually on control points; control points are not integrated, they are driven by `set_control_points`).
- **Key inputs:**
  - `x`, `v`: object vertex positions and velocities (Warp `vec3` arrays).
  - `control_x`, `control_v`: control point positions and velocities.
  - `springs`: `wp.array(dtype=wp.vec2i)` — each element is `(idx1, idx2)` of the two endpoints of a spring. Object vertices use indices `< num_object_points`; control points use indices `>= num_object_points` (stored as `num_object_points + control_index`).
  - `rest_lengths`: rest length per spring.
  - `spring_Y`: per-spring stiffness (stored in log space; used as `exp(spring_Y)` with clamping).
  - `dashpot_damping`, `spring_Y_min`, `spring_Y_max`.
- **Output:** `f`: force array (accumulated with atomics for object vertices).
- **Physics:** For each spring, force is along the direction between the two endpoints; stiffness term proportional to `(current_length/rest_length - 1)`; dashpot term from relative velocity along the spring axis. Forces on object vertices are added/subtracted; control points are not updated by the integrator.

### 3.3 `update_vel_from_force` (lines 161–183)

- **Role:** Update velocity from force, gravity, and drag.
- **Inputs:** current `v`, forces `f`, `masses`, `dt`, `drag_damping`, `reverse_factor` (for gravity direction).
- **Output:** `v_new`.
- **Formula:** `a = (f + m*g)/m`, `v_new = (v + a*dt) * exp(-dt*drag_damping)`.

### 3.4 `integrate_ground_collision` (lines 322–369)

- **Role:** Integrate position and velocity with ground plane collision (z = 0, with elasticity/friction).
- **Inputs:** current `x`, `v`, ground `collide_elas`, `collide_fric`, `dt`, `reverse_factor`.
- **Output:** `x_new`, `v_new` (including reflection and friction at ground).

So the **per-substep sequence** is: set control positions → eval springs → update velocities from forces → (optional object collision) → ground collision + integration.

---

## 4. Data structures

### 4.1 `State` (lines 34–64 in `spring_mass_warp.py`)

One `State` holds the **per-substep** simulation state (all Warp arrays on GPU):

- `wp_x`: vertex positions (object only).
- `wp_v`: vertex velocities (object only).
- `wp_v_before_collision`, `wp_v_before_ground`: intermediate velocities (after forces, after object collision).
- `wp_vertice_forces`: accumulated forces (cleared each substep).
- `wp_control_x`, `wp_control_v`: control point positions and velocities for this substep.

`SpringMassSystemWarp` keeps a **list** of states: `wp_states[0]` … `wp_states[num_substeps]`. After `step()`, `wp_states[0]` is the start of the frame and `wp_states[-1]` is the end (used as the new initial state for the next frame).

### 4.2 Spring-mass topology and parameters (in `SpringMassSystemWarp`)

- **Vertices:** `init_vertices` (and `wp_init_vertices`) — shape `(n_vertices, 3)`. First `num_object_points` are the deformable object; the rest are **control points** (only used as spring endpoints, not integrated).
- **Springs:** `init_springs` → `wp_springs`, dtype `vec2i`. Each row is `(i, j)`; if `i` or `j >= num_object_points`, that endpoint is a control point (index in `wp_control_x` is `i - num_object_points` or `j - num_object_points`).
- **Rest lengths:** `wp_rest_lengths`, one float per spring.
- **Stiffness:** `wp_spring_Y`, one value per spring (log-stiffness); used as `exp(spring_Y)` in `eval_springs`.
- **Masses:** `wp_masses`, one per **object** vertex (length `num_object_points`).

So: **nodes** = object vertices + control points (conceptually); **springs** = edges in this graph; **data structures** for the graph are `wp_springs`, `wp_rest_lengths`, `wp_spring_Y`, plus position/velocity arrays above.

### 4.3 Control points (actions) at the API level

- **Training / recorded data:** `controller_points` is a **torch** tensor of shape `(num_frames, num_control_points, 3)` (world-space positions per frame). It comes from the dataset (e.g. `RealData`) loaded from `final_data.pkl`.
- **Simulator internal:** The simulator only needs **two** frames at a time for one `step()`:
  - `wp_original_control_point`: control positions at **start** of the current frame.
  - `wp_target_control_point`: control positions at **end** of the current frame.
- So “actions” are **target control-point positions** (or equivalently, the delta from previous frame). The trainer sets these from `controller_points[frame_idx-1]` and `controller_points[frame_idx]`; the Gradio playground sets them from the current “previous” and “current” target state (see below).

---

## 5. Where data and “actions” come from: `final_data.pkl`

- **File:** e.g. `./data/different_types/<case_name>/final_data.pkl`.
- **Loaded by:** `src/qqtt/data/real_data.py` → `RealData` (used by `InvPhyTrainerWarp` in `trainer_warp.py`).

**Relevant keys (from `real_data.py` and data_process scripts):**

- `object_points`: shape `(num_frames, num_original_points, 3)` — tracked object surface (and similar) points per frame.
- `controller_points`: shape `(num_frames, num_controller_points, 3)` — **recorded “hand” or control positions** per frame (the “actions” in the training data).
- `object_visibilities`, `object_motions_valid`: per-point visibility and motion validity.
- `surface_points`, `interior_points`: extra surface and interior points used to build the full mesh/structure for the simulator.

**How this becomes the simulator:**

- `structure_points` = first-frame object + surface + interior points (used to build springs in `_init_start`).
- `controller_points` is passed as `controller_points=self.controller_points` into `SpringMassSystemWarp`. So the **same** tensor that holds per-frame control positions in the data is the source of “actions” for the simulator.
- For **training**, each frame pair `(frame_idx-1, frame_idx)` is used to set “original” and “target” control points and then run `step()` (see `set_controller_target` and the training loop in `trainer_warp.py`).

---

## 6. Gradio playground: how “actions” are recorded and applied

### 6.1 What counts as an “action” in the playground

- There is **no** separate “action log” file in the Gradio UI. “Actions” are **current target positions** of the control points, updated each simulation tick from **keyboard (or button) input**.
- So “actions” = the evolving **control target** over time: `current_target` (and the previous frame’s target `prev_target`).

### 6.2 Data flow (how keys become control targets and then forces)

1. **Key mapping** (`scripts/shims/interactive_playground_gradio.py` → `scripts/entrypoints/playground/`, and `trainer_warp.py`):
   - In `setup_simulation()`, `key_mappings` is set, e.g.:
     - `"w"` → `(0, [0.005, 0, 0])`  (panel 0, +X)
     - `"s"` → `(0, [-0.005, 0, 0])`
     - `"a"/"d"` → Y; `"q"/"e"` → Z.
     - For two control parts, `"i","k","j","l","o","u"` map to panel 1 with similar deltas.
   - `inv_ctrl` can flip the horizontal direction.

2. **Simulation thread** (`simulation_loop` in `scripts/entrypoints/playground/interactive_playground_gradio.py`):
   - Key press/release is sent via `control_queue` (`key:...` / `release:...`), and `trainer.pressed_keys` is updated.
   - Each loop iteration:
     - **Set control for this frame:**  
       `trainer.simulator.set_controller_interactive(prev_target, current_target)`  
       This copies `prev_target` → `wp_original_control_point` and `current_target` → `wp_target_control_point`.
     - **Run forward:**  
       `wp.capture_launch(self.trainer.simulator.forward_graph)`  
       That runs the captured `step()` (no gradients), so one frame of spring-mass with control moving from `prev_target` to `current_target`.
     - **Read new state:**  
       `x = wp.to_torch(self.trainer.simulator.wp_states[-1].wp_x)`  
       Then `set_init_state(wp_states[-1].wp_x, wp_states[-1].wp_v)` so the next frame continues from here.
     - **Update target from keys:**  
       `target_change = self.trainer.get_target_change()`  
       For each pressed key, `get_target_change()` sums the corresponding `(panel_id, delta)` into `target_change[panel_id]`. Then:
       - If two control parts: for each panel `i`, `current_target[masks_ctrl_pts[i]] += target_change[i]` (and same for hand positions).
       - If one part: `current_target += target_change`, and hand position is updated once.
     - **Advance “previous” for next iteration:**  
       `prev_target = current_target` (after the target update above, so next frame’s “original” is this frame’s “current”).

So the **data structure** for actions in the Gradio demo is:

- **In memory:** `current_target`, `prev_target` — tensors of shape `(num_control_points, 3)` (same layout as one frame of `controller_points`).
- **How they’re applied:** They are passed to `set_controller_interactive` and then used inside `step()` by `set_control_points` (interpolation) and `eval_springs` (springs between object vertices and these control positions).

There is no separate “action log” written by the Gradio UI; to “replay” the same sequence you would need to record `current_target` (or the key events) yourself and later call `set_controller_interactive(prev_target, current_target)` and `step()` in the same order.

### 6.3 Summary diagram (Gradio one frame)

```
Keys pressed → get_target_change() → target_change (n_ctrl_parts, 3)
       ↓
current_target += target_change   (and hand_left_pos / hand_right_pos)
       ↓
set_controller_interactive(prev_target, current_target)
       → wp_original_control_point = prev_target
       → wp_target_control_point  = current_target
       ↓
forward_graph (step())
       → set_control_points (interpolate within frame)
       → eval_springs (forces from object + control positions)
       → update_vel_from_force → (optional object_collision) → integrate_ground_collision
       ↓
wp_states[-1].wp_x, wp_states[-1].wp_v → new init state; render and motion interpolation
       ↓
prev_target = current_target (for next iteration)
```

---

## 7. Quick reference: important files and symbols

| What | Where |
|------|--------|
| Spring-mass forward step | `src/qqtt/model/diff_simulator/spring_mass_warp.py` → `SpringMassSystemWarp.step()` |
| Per-substep state | `State` in same file; list `simulator.wp_states` |
| Spring topology | `wp_springs` (vec2i), `wp_rest_lengths`, `wp_spring_Y` |
| Control interpolation | `set_control_points` kernel; driven by `wp_original_control_point`, `wp_target_control_point` |
| Data load | `src/qqtt/data/real_data.py` → `final_data.pkl` → `object_points`, `controller_points`, etc. |
| Trainer builds simulator | `src/qqtt/engine/trainer_warp.py` → `InvPhyTrainerWarp.__init__` → `SpringMassSystemWarp(...)` |
| Set control from data (training) | `simulator.set_controller_target(frame_idx)` |
| Set control for interactive (Gradio) | `simulator.set_controller_interactive(prev_target, current_target)` |
| Key → target delta (Gradio) | `trainer.key_mappings`, `trainer.get_target_change()` |
| Gradio simulation loop | `scripts/entrypoints/playground/interactive_playground_gradio.py` → `GradioInteractivePlayground.simulation_loop` |

---

## 8. For a future Rerun visualization

- **Nodes:** Object vertices = `wp_states[-1].wp_x` (or any `wp_states[i].wp_x`); you can add control positions from `wp_original_control_point` / `wp_target_control_point` (or the interpolated `wp_control_x` if you read it from inside a substep).
- **Edges:** Spring connectivity from `wp_springs` (and optionally `wp_rest_lengths` for length, `wp_spring_Y` for stiffness color-coding).
- **Replaying a sequence:** Either record `(prev_target, current_target)` each tick and replay with `set_controller_interactive` + `step()`, or record key events and replay the same key logic to recompute `current_target` and then run the same forward.

This should give you a clear map of the spring-mass model, data structures, and how actions are structured and applied for both training data and the Gradio demo.
