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
    outfiles = [
        '{}/{}.jpeg'.format(args.outdir, this_filename[:-3]),
        '{}/{}.pdf'.format(args.outdir, this_filename[:-3])]
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
                reportdir = os.path.join(indir, configdir, rundir, 'report')
                y = {}
                for file in os.listdir(reportdir):
                    if 'worker' not in file:
                        continue
                    data = np.genfromtxt(os.path.join(reportdir, file),
                                         dtype=str, delimiter='\t')
                    h, data = list(data[0]), data[1:]
                    for phase in locyc['phases']:
                        if phase not in y:
                            y[phase] = []
                        if phase == 'total':
                            acc = np.sum([float(r[h.index('total_time(sec)')])
                                          for r in data if ':' in r[h.index('function')]])
                        else:
                            acc = np.sum([float(r[h.index('total_time(sec)')])
                                          for r in data if phase in r[h.index('function')]])

                        y[phase].append(acc)
                for phase in y.keys():
                    if phase not in datadic[testcase][workers]:
                        datadic[testcase][workers][phase] = {}
                    datadic[testcase][workers][phase][run] = y[phase]
                run += 1
    figconf = locyc['figure']
    fig, ax = plt.subplots(**figconf['figure'])
    labels = set()
    phase = 'total'
    pos = 0
    w = 1 / (len(datadic.keys()) + 1)
    workers = [2, 4, 8, 12, 16]
    for testcase, dic in datadic.items():
        x = sorted(list(dic.keys()))
        for i, num_workers in enumerate(workers):
            if num_workers not in dic:
                continue
            # totavg = np.mean([v for v in datadic[num_workers]['total'].values()])
            total_time = np.max(
                [v for v in dic[num_workers]['total'].values()], axis=1)
            total_time_lb = np.min([v for v in dic[num_workers]['comm'].values()], axis=1) \
                + np.mean([v for v in dic[num_workers]['comp'].values()], axis=1) \
                + np.mean([v for v in dic[num_workers]
                          ['serial'].values()], axis=1)
            time_diff = 100 * (total_time - total_time_lb) / total_time
            avg = np.mean(time_diff)

            # vals = np.array([v for v in datadic[num_workers][phase].values()])
            # vals = vals.flatten() / totavg
            # vals = np.mean(vals, axis=1) / totavg
            # yerrlow = np.min(time_diff)
            # yerrhigh = np.max(time_diff)

            # label = None
            # if i == 0:
            #     label = 'measurement'

            # plt.errorbar([i+pos]*len(avgs), avgs/avg,
            plt.errorbar([i+pos]*len(time_diff), time_diff,
                         markersize=6, color='black',
                         marker='x', label=None,
                         linestyle='')
            label = testcase
            if label in labels:
                label = None
            labels.add(label)
            # plt.bar(i+pos, 1, width=w, yerr=[[yerrlow], [yerrhigh]],
            plt.bar(i+pos, avg, width=w,
                    # yerr=[[avg-yerrlow], [yerrhigh-avg]],
                    color=figconf['colors'][testcase],
                    edgecolor='0', label=label,
                    capsize=5, alpha=0.7)
            ax.annotate('{:.1f}'.format(avg), xy=(i+pos, avg),
                        textcoords='data', size=10,
                        ha='center', va='bottom',
                        color='black')
        pos += w

    ax.tick_params(**figconf['tick_params'])
    # plt.ylim(upper=2)
    plt.title('Time lost due to load imbalance, {}'.format(testcase))
    plt.xlabel('Cores (x10)', **figconf['title'])
    plt.ylabel('Normalized Runtime', **figconf['title'])
    plt.legend(**figconf['legend'])
    plt.xticks(np.arange(len(workers))+(pos-w)/2, workers, **figconf['title'])
    plt.yticks(fontsize=8)
    plt.tight_layout()

    for outfile in outfiles:
        save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
