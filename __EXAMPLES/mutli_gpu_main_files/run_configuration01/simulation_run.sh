#!/bin/bash
# Glue code to connect a python file with HTCondor arguments and point to virtual environment

# Activate the virtual environment
source "/afs/cern.ch/work/s/slauber/SimulationProjects/BLonD/.venv/bin/activate"

# Runs simulation ./blond_simulation_template/main
# You might want to adapt these lines
python3 "/afs/cern.ch/work/s/slauber/SimulationProjects/BLonD/EX_01_Acceleration.py"

