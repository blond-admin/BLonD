import os

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

blond_home = os.path.abspath(os.path.join(this_directory, '../../'))
exe_home = os.path.join(blond_home, '__EXAMPLES/gpu_main_files')
# batch_script = os.path.join(blond_home, 'scripts/other/batch-simple.sh')

mpirun = 'mpirun'
python = 'python'

cores_per_cpu = 20
# SLURM flags
slurm = {
    'script': os.path.join(blond_home, 'scripts/other/batch-simple.sh'),
    'submit': 'sbatch',
    'run': 'srun',
    'nodes': '--nodes',
    'workers': '--ntasks',
    'tasks_per_node': '--ntasks-per-node',
    'cores': '--cpus-per-task',
    'time': '-t',
    'output': '-o',
    'error': '-e',
    'jobname': '-J',

    'default_args': [
            '--mem', '56G',
            '--export', 'ALL',
            '--overcommit'
            # '--hint', 'nomultithread',
            # '--partition', 'inf-short'
    ]
}

# HTCondor args
condor = {
    'submit': 'condor_submit',
    'script': os.path.join(blond_home, 'scripts/other/condor.sub'),
    'executable': 'executable='+os.path.join(blond_home, 'scripts/other/condor.sh'),
    'output': 'output=',
    'error': 'error=',
    'log': 'log=',
    'arguments': 'arguments=',
    'jobname': '-batch-name',
    'time': '+MaxRuntime=',
    'cores': 'request_cpus=',
    'default_args': [
        # 'environment="PATH={PATH} PYTHONPATH={PYTHONPATH}"',
        # 'getenv=True',
        # 'should_transfer_files=IF_NEEDED'
    ]

}
