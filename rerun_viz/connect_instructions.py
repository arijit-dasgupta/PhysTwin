"""Printed help for Rerun live streaming from a remote VM."""

# Connect mode: Python pushes to viewer (JTAP-style). Needs SSH *remote* forward -R.
JTAP_STYLE_SSH = """
================================================================================
Rerun "connect" mode (same as JTAP: rr.init + rr.connect_grpc)
================================================================================
The Python process on the VM *pushes* logs TO the viewer. The viewer must be
reachable from the VM at the URL passed to connect_grpc (default:
rerun+http://127.0.0.1:9876/proxy).

Typical setup (viewer on your laptop, code on VM):

  1) On your LAPTOP, start the viewer first (leave it running):
       rerun

  2) SSH from laptop to VM with a *remote* forward so that traffic to the
     VM's localhost:9876 is sent to your laptop's localhost:9876 (where the
     viewer listens):

       ssh -R 9876:127.0.0.1:9876 user@vm

     (VS Code "Remote" can add a similar Remote port forward: bind 9876 on
     remote host -> 9876 on local.)

  3) On the VM, run this script. It will call rr.connect_grpc() which connects
     to 127.0.0.1:9876 *on the VM* — which the SSH -R tunnel sends to the viewer.

If you use a different port, change 9876 everywhere (SSH -R, viewer, and
--connect-url if you pass one).

**If this flakes (flush timeouts / empty viewer):** use **serve** mode instead
(`--rerun_mode serve`). See REMOTE_LIVE_SERVE_SSH in connect_instructions.py.
================================================================================
"""

# Serve mode: recommended for VM → Rerun app on laptop. Needs SSH *local* forward -L.
REMOTE_LIVE_SERVE_SSH = """
================================================================================
Rerun LIVE (recommended VM → laptop): serve_grpc on VM, viewer on laptop
================================================================================
Python on the VM **listens**; the Rerun app on your laptop **connects** through SSH.

  1) SSH from LAPTOP to VM with a *local* forward (NOT -R for the same port):

       ssh -L 9876:127.0.0.1:9876 user@your-vm

     Or in ~/.ssh/config under your Host:

       LocalForward 9876 127.0.0.1:9876

     Comment out **RemoteForward 9876** if present — you cannot mix -R and -L
     on the same port in one session.

  2) On the **VM**, run scripts with **serve** (default for our CLIs):

       python -m rerun_viz.jtap_style_demo --mode serve
       python -m rerun_viz.replay_recorded ... --rerun_mode serve

  3) On the **LAPTOP** (separate terminal), connect the viewer **while the VM script
     is still running** (or use a script that keeps the server alive after logging):

       rerun rerun+http://127.0.0.1:9876/proxy

     If the Python process on the VM has already exited, **nothing is listening**
     and the viewer will be empty / fail to connect.

Match viewer version to SDK (e.g. rerun 0.30.x ↔ rerun_py 0.30.x).

**WEB VIEWER (if native app fails):** Add LocalForward 9090 127.0.0.1:9090,
run scripts that use serve_web_viewer, then open http://127.0.0.1:9090 in a browser.
================================================================================
"""

# Web viewer: serve HTML in browser instead of native app. Needs -L for both 9876 and 9090.
WEB_VIEWER_SSH = """
  LocalForward 9876 127.0.0.1:9876
  LocalForward 9090 127.0.0.1:9090
  Then open in BROWSER:  http://127.0.0.1:9090
"""
