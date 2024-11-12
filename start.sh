#!/bin/bash

# Unified script to start Jupyter Lab or Gradio
if [ "$1" == "jupyter" ]; then
    echo "Starting Jupyter Lab..."
    jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
elif [ "$1" == "gradio" ]; then
    echo "Starting Gradio App..."
    python gradio_app.py
else
    echo "Usage: $0 [jupyter|gradio]"
    echo "  jupyter: Start Jupyter Lab"
    echo "  gradio: Start Gradio App"
fi
