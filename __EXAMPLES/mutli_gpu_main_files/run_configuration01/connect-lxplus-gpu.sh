# log into cluster; change path to end up in the same folder that this script
# dont forget to replace /y/yusername/ by your initial and username
ssh lxplus-gpu.cern.ch -t 'cd /afs/cern.ch/work/y/yusername/SimulationProjects/BLonD/__EXAMPLES/mutli_gpu_main_files/run_configuration01/; bash'
# after logging in, you might want to
# 1. Update your project 'git pull'
# 2. Run a simulation 'run_configuration01/simulation_run.sh'
source /afs/cern.ch/work/y/yusername/SimulationProjects/BLonD/.venv/bin/activate
python -c "import cupy; cupy.ones(10)"
