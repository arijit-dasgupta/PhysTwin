"""One forward physics step aligned with ``rerun_viz.replay_core`` (no Rerun)."""

from __future__ import annotations

import warp as wp

from qqtt.utils import cfg

# When ``cfg.use_graph`` is True, timing uses ``wp.capture_launch(forward_graph)`` (one outer step).
# Optional segment profiling in the benchmark forces ``cfg.use_graph = False`` for that call so
# ``SpringMassSystemWarp.step()`` runs eagerly; see ``benchmarks.spring_mass_gpu.profile_segments``.


def step_forward_once(simulator, frame_idx: int) -> None:
    """Single outer step: controller target → optional collision graph → graph or ``step()``."""
    simulator.set_controller_target(frame_idx, pure_inference=True)
    if simulator.object_collision_flag:
        simulator.update_collision_graph()
    if cfg.use_graph and hasattr(simulator, "forward_graph"):
        wp.capture_launch(simulator.forward_graph)
    else:
        simulator.step()


def step_and_advance_state(simulator, frame_idx: int) -> None:
    """Like one replay frame: step then carry state forward."""
    step_forward_once(simulator, frame_idx)
    simulator.set_init_state(
        simulator.wp_states[-1].wp_x,
        simulator.wp_states[-1].wp_v,
        pure_inference=True,
    )
