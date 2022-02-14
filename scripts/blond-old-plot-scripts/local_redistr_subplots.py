#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import matplotlib.patches as mpatches
import sys
from plot.plotting_utilities import *
from cycler import cycle
import yaml
import argparse
import csv
import re

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]


parser = argparse.ArgumentParser(
    description='The run-time percent that is lost due to load imbalance.',
    usage='{} infile -o outfile'.format(this_filename))

parser.add_argument('-i', '--indir', nargs='+',
                    help='The input directories that contains the raw data.')

parser.add_argument('-o', '--outdir', action='store', type=str,
                    default='./', help='The directory to store the plots.')

parser.add_argument('-yg', '--yamlglobal', type=str,
                    default=this_directory+'/configs/global_config.yml',
                    help='The global yaml config file.')

parser.add_argument('-yl', '--yamllocal', type=str,
                    default=this_directory + '/configs/'
                    + this_filename.replace('.py', '.yml'),
                    help='The local yaml config file.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots or save only.')

# parser.add_argument('-r', '--record', action='store_true',
#                     help='Store the output data in a file.')

re_log = '.*Turn\s(.*),\sTconst\s(.*),\sTcomp\s(.*),\sTcomm\s(.*),\sTsync\s(.*),\sLatency\s(.*),\sParticles\s(.*)'
re_log = re.compile(re_log)

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # read yaml config file
    yc = yaml.load(open(args.yamlglobal, 'r'))
    globyc = yc['global']
    locyc = yaml.load(open(args.yamllocal, 'r'))

    datadic = {}

    for indir in args.indir:

        if 'lhc' in indir.lower():
            testcase = 'LHC'
        elif 'sps' in indir.lower():
            testcase = 'SPS'
        elif 'ps' in indir.lower():
            testcase = 'PS'
        elif 'ex01' in indir.lower():
            testcase = 'EX01'

        for configdir in os.listdir(indir):
            if not os.path.isdir(os.path.join(indir, configdir)):
                continue

            parts = int(configdir.split('_p')[1].split('_')[0])
            bunches = int(configdir.split('_b')[1].split('_')[0])
            slices = int(configdir.split('_s')[1].split('_')[0])
            turns = int(configdir.split('_t')[1].split('_')[0])
            workers = int(configdir.split('_w')[1].split('_')[0])
            threads = int(configdir.split('_o')[1].split('_')[0])
            nodes = int(configdir.split('_N')[1].split('_')[0])
            red = int(configdir.split('_red')[1].split('_')[0])
            seed = int(configdir.split('_seed')[1].split('_')[0])
            approx = int(configdir.split('_approx')[1].split('_')[0])
            mpiv = configdir.split('_mpi')[1].split('_')[0]
            lb = configdir.split('_lb')[1].split('_')[0]
            lba = configdir.split('_lba')[1].split('_')[0]

            # if bunches not in locyc['keep_only']['bunches'] and \
            #         turns not in locyc['keep_only']['turns']:
            #     continue

            conf = '{}_{}w_{}Mp_{}Kt_approx{}'.format(
                testcase, workers, int(parts * bunches/1e6), int(turns/1000), approx)
            if conf not in datadic:
                datadic[conf] = {}

            if workers not in datadic[conf]:
                datadic[conf][workers] = {}
            if lb not in datadic[conf][workers]:
                datadic[conf][workers][lb] = {}

            for run, rundir in enumerate(os.listdir(os.path.join(indir, configdir))):
                if not os.path.isdir(os.path.join(indir, configdir, rundir)):
                    continue
                # datadic[conf][workers][lb][run] = {}
                reportdir = os.path.join(indir, configdir, rundir, 'log')
                for file in os.listdir(reportdir):
                    if 'worker' not in file:
                        continue
                    worker = int(file.split('worker-')[1].split('.log')[0])
                    if worker not in datadic[conf][workers][lb]:
                        datadic[conf][workers][lb][worker] = {
                            'turn': [], 'tconst': [], 'tcomp': [], 'tcomm': [],
                            'tsync': [], 'latency': [], 'particles': [], 'total': [],
                            'tcomp+tconst': []}

                    y = {'turn': [], 'tconst': [], 'tcomp': [], 'tcomm': [],
                         'tsync': [], 'latency': [], 'particles': [],
                         'tcomp+tconst': []}
                    for line in open(os.path.join(reportdir, file), 'r'):
                        match = re_log.search(line)
                        if not match:
                            continue
                        turn, tconst, tcomp, tcomm, tsync, latency, particles = match.groups()
                        y['turn'].append(int(turn))
                        y['tconst'].append(float(tconst))
                        y['tcomp'].append(float(tcomp))
                        y['tcomm'].append(float(tcomm))
                        y['tsync'].append(float(tsync))
                        y['latency'].append(float(latency))
                        y['particles'].append(float(particles))
                    y['total'] = np.array(y['tconst']) + np.array(y['tsync']) \
                        + np.array(y['tcomp']) + np.array(y['tcomm'])
                    y['tcomp+tconst'] = np.array(y['tconst']) + \
                        np.array(y['tcomp'])

                    for key in y.keys():
                        datadic[conf][workers][lb][worker][key].append(
                            y[key])

    figconf = locyc['figure']

    for tc in datadic.keys():
        for w in datadic[tc].keys():
            fig, ax_arr = plt.subplots(**figconf['figure'])
            ax_arr = ax_arr.flatten()
            outfiles = [
                '{}/{}-{}-{}.jpeg'.format(args.outdir,
                                          this_filename[:-3], tc, w),
                '{}/{}-{}-{}.pdf'.format(args.outdir, this_filename[:-3], tc, w)]
            fig.suptitle('{}'.format(tc), **figconf['suptitle'])
            for i, key in enumerate(locyc['to_plot']):
                ax = ax_arr[i]
                plt.sca(ax)
                labels = set()
                plt.grid(**figconf['grid'])
                for version in datadic[tc][w].keys():
                    # for run in datadic[tc][w][version].keys():
                    for wid, vals in datadic[tc][w][version].items():
                        x = np.array(vals['turn'], dtype=int)
                        x = np.mean(x, axis=0) / 1000.
                        y = np.mean(
                            np.array(vals[key], dtype=float), axis=0)/1000
                        c = '{}-{}'.format(version, wid)
                        label = None
                        if key == 'total':
                            if wid == 0:
                                label = 'Total: {:.1f}s'.format(np.sum(y))
                        elif key == 'tcomp+tconst':
                            if wid == 0:
                                label = 'Total: {:.1f}s'.format(np.sum(y))
                        elif c not in labels:
                                labels.add(c)
                                label = figconf['labels'][c]
                        plt.plot(x[::2], y[::2], color=figconf['colors'][c], label=label,
                                 marker=figconf['markers'][version], lw=1, markersize=6)
                        # if key == 'total' and wid == 1:
                        #     # x = x[-1]
                        #     # y = np.sum(y)
                        #     ax.annotate('Total: {:.1f} s'.format(np.sum(y)), xy=(x[len(x)//2], y[-1]/2),
                        #                 textcoords='data', ha='center', va='center',
                        #                 color = figconf['colors'][c],
                        #                 **figconf['annotate'])

                ax.tick_params(**figconf['tick_params'])
                plt.title(key, **figconf['title'])
                if i > 2:
                    plt.xlabel('turn (k)', **figconf['label'])
                if i == 0 or i == 3:
                    plt.ylabel('time (s)', **figconf['label'])
                # if i == 1 or i==3 or i==4 or i == 5:
                plt.legend(**figconf['legend'])
                # plt.xticks(np.arange(len(workers))+(pos-w)/2, workers, **figconf['title'])
                plt.yticks(**figconf['ticks']['y'])
                plt.xticks(**figconf['ticks']['x'])
                plt.tight_layout()

            plt.subplots_adjust(**figconf['subplots_adjust'])

            for outfile in outfiles:
                save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
            if args.show:
                plt.show()
            plt.close()

# label = testcase
# if label in labels:
#     label = None
# labels.add(label)

# ax.annotate('{:.1f}'.format(avg), xy=(i+pos, avg),
#             textcoords='data', size=10,
#             ha='center', va='bottom',
#             color='black')
