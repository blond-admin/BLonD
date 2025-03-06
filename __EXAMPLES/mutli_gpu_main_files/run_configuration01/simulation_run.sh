#!/bin/bash
# Glue code to connect a python file with HTCondor arguments and point to virtual environment

# Calculate the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"


# Activate the virtual environment
source "$SCRIPT_DIR/../../../.venv/bin/activate"

# Runs simulation ./blond_simulation_template/main
# You might want to adapt these lines
python3 "$SCRIPT_DIR/EX_01_Acceleration.py"
