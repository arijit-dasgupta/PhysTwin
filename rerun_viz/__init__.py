"""
Rerun visualization for PhysTwin spring–mass simulations.

Main entry points:
- ``replay_recorded``: CLI wrapper; core logic in ``replay_core``.
- ``replay_core``: load case, replay, stream to Rerun.
- ``spring_mass_logging``: helpers to log simulator state to Rerun.
- ``inspect_case_stats``: inspect mass/stiffness for a case.
- ``case_setup``: shared yaml + optimal-params loading.
- ``port_util`` / ``connect_instructions``: remote VM + Rerun workflows.
"""
