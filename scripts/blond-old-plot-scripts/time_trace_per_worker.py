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
                    default=this_directory+'global_config.yml',
                    help='The global yaml config file.')

parser.add_argument('-yl', '--yamllocal', type=str,
                    default=this_directory +
                    this_filename.replace('.py', '.yml'),
                    help='The local yaml config file.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots or save only.')

# parser.add_argument('-r', '--record', action='store_true',
#                     help='Store the output data in a file.')

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

        if testcase not in datadic:
            datadic[testcase] = {}

        for configdir in os.listdir(indir):
            if not os.path.isdir(os.path.join(indir, configdir)):
                continue
            parts = configdir.split('_p')[1].split('_')[0]
            bunches = configdir.split('_b')[1].split('_')[0]
            slices = configdir.split('_s')[1].split('_')[0]
            turns = configdir.split('_t')[1].split('_')[0]
            workers = int(configdir.split('_w')[1].split('_')[0])
            threads = configdir.split('_o')[1].split('_')[0]
            nodes = configdir.split('_N')[1].split('_')[0]
            red = configdir.split('_r')[1].split('_')[0]
            seed = configdir.split('_seed')[1].split('_')[0]
            approx = configdir.split('_approx')[1].split('_')[0]
            # mpiv = configdir.split('_mpi')[1].split('_')[0]
            if workers not in datadic:
                datadic[testcase][workers] = {}

            run = 0
            for rundir in os.listdir(os.path.join(indir, configdir)):
                if not os.path.isdir(os.path.join(indir, configdir, rundir)):
                    continue
                file = os.path.join(indir, configdir, rundir, 'particles-workers.csv')
                if not os.path.exists(file):
                    continue
                datadic[testcase][workers][run] = {}
                data = np.genfromtxt(file, dtype=str, delimiter='\t')
                h, data = list(data[0]), data[1:]
                # h = wid, turn_num, parts, tcomp, tconst, tcomm, tpp
                for r in data:
                    wid = int(r[0])
                    datadic[testcase][workers][run][wid] = {}
                    for i, j in zip(h, r):
                        datadic[testcase][workers][run][wid][i] = j.split('|')
                run += 1
    # phase = 'total'
    # pos = 0
    # w = 1 / (len(datadic.keys()) + 1)
    workers = locyc['workers']
    phases = locyc['phases']
    for tc, dic in datadic.items():
        outfiles = [
            '{}/{}-{}.jpeg'.format(args.outdir, this_filename[:-3], tc),
            '{}/{}-{}.pdf'.format(args.outdir, this_filename[:-3], tc)]
        figconf = locyc['figure']
        fig, ax_arr = plt.subplots(**figconf['figure'])
        # axes = axes.flat
        i = 0
        labels = set()
        # x = sorted(list(dic.keys()))
        for i, num_workers in enumerate(workers):
            for j, phase in enumerate(phases):
                ax = ax_arr[i][j]
                plt.sca(ax)
                if i == 0:
                    plt.title(phase, **figconf['title'])
                if j == 0:
                    plt.ylabel('W={}'.format(num_workers), **figconf['title'])
                if i == len(workers)-1:
                    plt.xlabel('Turn', **figconf['title'])
                for run in dic[num_workers].keys():
                    for wid, vals in dic[num_workers][run].items():
                        t = np.array(vals['turn_num'], int)
                        dt = np.diff(np.array(['0'] + vals['turn_num'], int))
                        y = np.array(vals[phase], float) / dt
                        plt.plot(t, y, color=figconf['colors'][2*run+1], 
                            lw=1.5)
                # plt.legend(**figconf['legend'])
                # plt.xticks(np.arange(len(workers))+(pos-w)/2, workers, **figconf['title'])
                # plt.yticks(fontsize=8)
                ax.tick_params(**figconf['tick_params'])

        #     # label = None
        #     # if i == 0:
        #     #     label = 'measurement'

        #     # plt.errorbar([i+pos]*len(avgs), avgs/avg,
        #     plt.errorbar([i+pos]*len(time_diff), time_diff,
        #                  markersize=6, color='black',
        #                  marker='x', label=None,
        #                  linestyle='')
        #     label = testcase
        #     if label in labels:
        #         label = None
        #     labels.add(label)
        #     # plt.bar(i+pos, 1, width=w, yerr=[[yerrlow], [yerrhigh]],
        #     plt.bar(i+pos, avg, width=w,
        #             # yerr=[[avg-yerrlow], [yerrhigh-avg]],
        #             color=figconf['colors'][testcase],
        #             edgecolor='0', label=label,
        #             capsize=5, alpha=0.7)
        #     ax.annotate('{:.1f}'.format(avg), xy=(i+pos, avg),
        #                 textcoords='data', size=10,
        #                 ha='center', va='bottom',
        #                 color='black')
        # pos += w

        # plt.ylim(upper=2)
        plt.tight_layout()
        plt.subplots_adjust(**figconf['subplots_adjust'])

        for outfile in outfiles:
            save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
