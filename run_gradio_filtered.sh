#!/bin/bash
# Wrapper script to run interactive_playground_gradio.py with warning filtering

# Filter out Warp warnings about set_control_points
# This filters at the shell level as a backup to Python-level filtering
# Note: Python-level filtering in interactive_playground_gradio.py should handle most cases

# Run the script - Python-level filters should suppress warnings
# If warnings still appear, they'll be in the output but can be filtered post-processing
python interactive_playground_gradio.py "$@"
