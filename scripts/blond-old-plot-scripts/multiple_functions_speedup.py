import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from itertools import cycle
import matplotlib.ticker
import sys
import argparse

from plot.plotting_utilities import *

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

parser = argparse.ArgumentParser(description='Run MPI jobs locally.',
                                 usage='python local_scan_mpi.py -i in.yml')

parser.add_argument('-c', '--case', type=str, choices=['lhc', 'sps', 'ps', 'ex01'],
                    help='The test-case to plot.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')

args = parser.parse_args()


project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/mpi_scalability/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

case = args.case

config = {
    # 'figures': {
    #     '{} DLB'.format(case.upper()): {
    'base': '{}/local_raw/{}/mpich3/avg-report.csv'.format(res_dir, case.upper()),
    'files': [
            '{}/raw/{}/mvapich2/avg-report.csv'.format(
                res_dir, case.upper()),
            # '{}/raw/{}/mpich3/avg-report.csv'.format(
            #     res_dir, case.upper()),
            '{}/raw/{}/lb-mvapich2/avg-report.csv'.format(
                res_dir, case.upper()),
            # '{}/raw/{}/lb-mpich3/avg-report.csv'.format(
            #     res_dir, case.upper()),
            '{}/raw/{}/tp-mvapich2/avg-report.csv'.format(
                res_dir, case.upper()),
            # '{}/raw/{}/tp-mpich3/avg-report.csv'.format(
            #     res_dir, case.upper()),
            '{}/raw/{}/lb-tp-mvapich2/avg-report.csv'.format(
                res_dir, case.upper()),
            # '{}/raw/{}/lb-tp-mpich3/avg-report.csv'.format(
            #     res_dir, case.upper()),
    ],
    'lines': {
        'mpi': ['mpich3', 'mvapich2'],
        'lb': ['interval', 'reportonly', 'off'],
        'approx': ['0', '2'],
        'lba': ['500'],
        'b': ['96', '72', '21'],
        't': ['5000', '1000'],
        'n': ['1', '4', '8', '12', '16']
        # 'type': ['total'],
    },
    'outfiles': ['{}/{}-{}-{}.pdf', '{}/{}-{}-{}.jpg'],
    # },
    # '{} DLB \w TP'.format(case.upper()): {
    #     'files': [
    #         '{}/raw/{}/tp-mvapich2/comm-comp-report.csv'.format(
    #             res_dir, case.upper()),
    #         '{}/raw/{}/lb-tp-mvapich2/comm-comp-report.csv'.format(
    #             res_dir, case.upper()),
    #         '{}/raw/{}/tp-mpich3/comm-comp-report.csv'.format(
    #             res_dir, case.upper()),
    #         '{}/raw/{}/lb-tp-mpich3/comm-comp-report.csv'.format(
    #             res_dir, case.upper())
    #     ],
    #     'lines': {
    #         'mpi': ['mpich3', 'mvapich2'],
    #         'lb': ['interval', 'reportonly'],
    #         'approx': ['0', '2'],
    #         'lba': ['100', '500'],
    #         'b': ['96', '72', '21'],
    #         't': ['5000'],
    #         'type': ['total'],
    #     },
    #     'outfiles': ['{}/{}-DLB-TP.pdf'.format(images_dir, case)]
    # },
    # },
    'markers': {
        'ex01': 'd',
        'lhc': 'o',
        'sps': 's',
        'ps': 'x'
    },
    'ls': {
        'ex01': '-:',
        'lhc': '-',
        'sps': ':',
        'ps': '--'
    },
    'colors': ['0.95', '0.7', '0.4', '0.1'],
    'hatches': ['', '', '', ''],
    'reference': {
        # 'ex01': {'time': 21.4, 'ppb': 1000000, 'turns': 2000},
        # 'sps': {'time': 430., 'ppb': 4000000, 'turns': 100},
        # 'sps': {'ppb': 4000000, 'b': 72, 'turns': 500, 'w': 1,
        #         'omp': 1, 'time': 1497.8},
        'sps': {'ppb': 4000000, 'b': 72, 'turns': 1000, 'w': 1,
                'omp': 10, 'time': 415.4},
        # 'lhc': {'time': 2120., 'ppb': 2000000, 'turns': 1000},
        # 'lhc': {'ppb': 2000000, 'b': 96, 'turns': 500, 'w': 1,
        #         'omp': 1, 'time': 681.59},
        'lhc': {'ppb': 2000000, 'b': 96, 'turns': 1000, 'w': 1,
                'omp': 10, 'time': 177.585},

        # 'ps': {'time': 1623.7, 'ppb': 4000000, 'turns': 2000},
        # 'ps': {'time': 466.085, 'ppb': 8000000, 'b': 21, 'turns': 500},
        # 'ps': {'ppb': 8000000, 'b': 21, 'turns': 500, 'w': 1,
        #        'omp': 1, 'time': 502.88},
        'ps': {'ppb': 8000000, 'b': 21, 'turns': 1000, 'w': 1,
               'omp': 10, 'time': 142.066},
    },
    # 'sequence': ['mpich3']

    # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    'x_name': 'n',
    'x_to_keep': [4, 8, 12, 16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    # 'y_err_name': 'std',
    'xlabel': ['', 'Phase'],
    'ylabel': ['Speedup', 'Runtime %'],
    'title': {
        'fontsize': 10,
        'y': 0.88,
        'x': 0.55,
        'fontweight': 'bold',
    },
    'subplots': {
        # 'figsize': (8, 6),
        'nrows': 2,
        'ncols': 1,
        'sharex': True
    },
    'annotate': {
        'fontsize': 9,
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 9},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 2, 'handlelength': 2, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.5,
    },
    'subplots_adjust': {
        'wspace': 0.05, 'hspace': 0.1
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'inout', 'length': 3, 'width': 0.5,
    },
    'ylim': [[0, 14], [0, 100]],

}

if __name__ == '__main__':
    # first read the base
    data = np.genfromtxt(config['base'], delimiter='\t', dtype=str)
    header, data = list(data[0]), data[1:]
    temp = get_plots(header, data, config['lines'],
                     exclude=config.get('exclude', []),
                     prefix=True)
    # Convert to convenient form
    basedic = {}
    for key, arr in temp.items():
        for row in arr:
            fname = row[header.index('function')]
            if 'other' in fname.lower():
                continue
            category = fname.split(':')[0]
            turns = float(row[header.index('t')])
            if category not in basedic:
                basedic[category] = {'time': 0, 'percent': 0, 'turns': turns}
            basedic[category]['time'] += float(
                row[header.index('total_time(sec)')])
            basedic[category]['percent'] += float(
                row[header.index('global_percentage')])

    for file in config['files']:
        # We have two subplots, one showing the speedup and the other the
        # time breakdown.

        indic = {}
        # first read the file
        data = np.genfromtxt(file, delimiter='\t', dtype=str)
        header, data = list(data[0]), data[1:]
        temp = get_plots(header, data, config['lines'],
                         exclude=config.get('exclude', []),
                         prefix=True)
        for key in temp.keys():
            indic['_{}'.format(key)] = temp[key].copy()

        # Now convert it into the format we want:
        # for each key, a dict with func_name, time, percentage, speedup

        plotdic = {}
        for k, arr in indic.items():
            key = k.split('_n')[0]
            workers = int(k.split('_n')[1].split('_')[0])
            if key not in plotdic:
                plotdic[key] = {}
            # if workers not in plotdic:
            #     plotdic[key][workers] = {'time': [], 'fname': [],
            #                              'percent': [], 'speedup': []}

            for row in arr:
                fname = row[header.index('function')]
                if 'other' in fname.lower():
                    continue
                category = fname.split(':')[0]
                turns = float(row[header.index('t')])

                if category not in plotdic[key]:
                    plotdic[key][category] = {'time': [], 'workers': [],
                                              'percent': [], 'turns': []}
                time = float(row[header.index('total_time(sec)')])
                percent = float(row[header.index('global_percentage')])
                if workers in plotdic[key][category]['workers']:
                    idx = plotdic[key][category]['workers'].index(workers)
                    plotdic[key][category]['time'][idx] += time
                    plotdic[key][category]['percent'][idx] += percent
                else:
                    plotdic[key][category]['time'].append(time)
                    plotdic[key][category]['workers'].append(workers)
                    plotdic[key][category]['percent'].append(percent)
                    plotdic[key][category]['turns'].append(turns)

            for cat, v in plotdic[key].items():
                if cat == 'comm':
                    plotdic[key][cat]['speedup'] = v['time'][0] \
                        / np.array(v['time'])
                else:
                    plotdic[key][cat]['speedup'] = np.array(basedic[cat]['time']) \
                        / np.array(basedic[cat]['turns']) \
                        / (np.array(v['time']) / np.array(v['turns']))

        # Okay now we can plot the speedup in one plot and
        # the percent in the other
        # bar plot, plot first all comp, then serial, then comm
        # Maybe three subplots? because of the different ranges

        # first the speedup in bar plot format
        for k in sorted(plotdic.keys()):
            fig, axes = plt.subplots(**config['subplots'])

            labels = set()
            mpiv = k.split('_mpi')[1].split('_')[0]
            lb = k.split('lb')[1].split('_')[0]
            lba = k.split('lba')[1].split('_')[0]
            approx = k.split('approx')[1].split('_')[0]
            if lb == 'interval':
                lb = 'LB'
            elif lb == 'reportonly':
                lb = 'NoLB'
            if approx == '2':
                approx = 'TP'
            elif approx == '0':
                approx = 'NoTP'

            for i in [0, 1]:
                ax = axes[i]
                plt.sca(ax)
                plt.grid(True, which='major', alpha=0.5)
                plt.grid(False, which='major', axis='x')
                plt.xlabel(config['xlabel'][i], fontsize=config['fontsize'])
                plt.ylabel(config['ylabel'][i], fontsize=config['fontsize'])
                # plt.xticks(fontsize=config['fontsize'])
                plt.yticks(**config['ticks'])
                ax.tick_params(**config['tick_params'])
                plt.ylim(config['ylim'][i])
                if i == 0:
                    plt.title('{}-{}-{}'.format(case.upper(), lb, approx),
                              **config['title'])

            pos = 0.
            step = 1.
            xticks = [[], []]
            for fname in sorted(plotdic[k].keys()):

                ax = axes[0]
                plt.sca(ax)

                vals = plotdic[k][fname]
                w = step / (len(vals['workers']) + 1)
                plt.bar(pos + np.arange(0, len(vals['workers'])) * w,
                        vals['speedup'], width=w,
                        edgecolor='black', color=config['colors'])
                if 'total_time' in fname:
                    annotate(ax, pos + np.arange(0, len(vals['workers'])) * w,
                             vals['speedup'], **config['annotate'])

                ax = axes[1]
                plt.sca(ax)
                plt.bar(pos + np.arange(0, len(vals['workers'])) * w,
                        vals['percent'], width=w,
                        edgecolor='black', color=config['colors'])
                xticks[0].append(pos)
                xticks[1].append(fname)
                pos += step
            handles = []
            for c, label in zip(config['colors'], vals['workers']):
                patch = mpatches.Patch(label=label, edgecolor='black',
                                       facecolor=c,
                                       linewidth=.5, alpha=0.9)
                handles.append(patch)
            plt.legend(handles=handles, **config['legend'])

            plt.xticks(xticks[0], xticks[1], fontsize=config['fontsize'],
                       rotation=0)

            plt.tight_layout()
            plt.subplots_adjust(**config['subplots_adjust'])
            for file in config['outfiles']:
                file = file.format(images_dir, case, lb, approx)
                save_and_crop(fig, file, dpi=600, bbox_inches='tight')
            if args.show:
                plt.show()
            plt.close()
