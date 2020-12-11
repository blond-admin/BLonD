import subprocess
import os
import sys
import argparse
import glob

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

parser = argparse.ArgumentParser(description='Plot all the extracted results.',
                                 usage='python {} -i results/local/ -t lhc sps ps'.format(this_filename[:-3]))


parser.add_argument('-i', '--indir', type=str, default='./results/local/',
                    help='The directory with the raw data. Default: ./results/local/')

parser.add_argument('-t', '--testcases', type=str, default='lhc,sps,ps',
                    help='A comma separated list of the testcases to run. Default: lhc,sps,ps')



plot_scripts = {
    'intermediate_effect_analysis.py': ['approx0-interm', 'approx1-interm',
                                        'approx2-interm', 'tp-approx0-interm',
                                        'lb-tp-approx0-interm',
                                        'lb-tp-approx1-interm',
                                        'lb-tp-approx2-interm'],
    'strong_scaling_experiment.py': ['lb-tp-approx0-strong-scaling',
                                     'lb-tp-approx1-strong-scaling',
                                     'lb-tp-approx2-strong-scaling'],
    'workers_per_node_evaluation.py': ['approx0-workers'],
    'load_imbalance_spread.py': ['approx0-spread'],
    'time_breakdown_strong_scaling.py': ['lb-tp-approx0-strong-scaling',
                                         'lb-tp-approx1-strong-scaling'],
    'mpi_implementations_bench.py': ['approx0-impl'],
    'weak_scaling_experiment.py': ['lb-tp-approx0-weak-scaling',
                                   'lb-tp-approx1-weak-scaling',
                                   'lb-tp-approx2-weak-scaling']
}

if __name__ == '__main__':
    args = parser.parse_args()

    failed_plots = []
    not_ready_plots = []
    for plot, requirements in plot_scripts.items():
        isReady = True
        for case in args.testcases.split(','):
            for req in requirements:
                directory = os.path.join(args.indir, case, req)
                if os.path.isdir(directory) and len(glob.glob(directory + '/.extracted')) > 0:
                    continue
                else:
                    isReady = False
                    break
        if isReady:
            cmd = ['python', os.path.join(this_directory, plot),
                   '-i', args.indir,
                   '-c', args.testcases]
            output = subprocess.run(cmd, stdout=sys.stdout,
                                    stderr=subprocess.STDOUT, env=os.environ.copy())
            if output.returncode != 0:
                failed_plots.append(plot)
        else:
            not_ready_plots.append(plot)

    if failed_plots:
        print('[{}] The following plots raised an error:'.format(
            this_filename[:-3]))
        for f in failed_plots:
            print(f)

    if not_ready_plots:
        print('[{}] The following plots could not find the required extracted results:'.format(
            this_filename[:-3]))
        for f in not_ready_plots:
            print(f)
