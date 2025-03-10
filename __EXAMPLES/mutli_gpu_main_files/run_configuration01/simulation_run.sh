#!/bin/bash
# Glue code to connect a python file with HTCondor arguments and point to virtual environment

# Activate the virtual environment
source "/afs/cern.ch/work/y/yusername/SimulationProjects/BLonD/.venv/bin/activate"

# Runs simulation ./blond_simulation_template/main
# You might want to adapt these lines
python "/afs/cern.ch/work/y/yusername/SimulationProjects/BLonD/__EXAMPLES/mutli_gpu_main_files/run_configuration01/EX_01_Acceleration.py"

