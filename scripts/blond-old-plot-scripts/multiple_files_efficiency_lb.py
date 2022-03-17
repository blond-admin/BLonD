import matplotlib.pyplot as plt
import numpy as np

import os
from matplotlib import cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from itertools import cycle
import matplotlib.ticker
import sys
from plot.plotting_utilities import *
import argparse

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
images_dir = res_dir + 'plots/redistribute/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

case = args.case

config = {
    'figures': {
        '{} DLB'.format(case.upper()): {
            'files': [
                '{}/raw/{}/mvapich2/comm-comp-report.csv'.format(
                    res_dir, case.upper()),
                '{}/raw/{}/tp-mvapich2/comm-comp-report.csv'.format(
                    res_dir, case.upper()),
                '{}/raw/{}/lb-mvapich2/comm-comp-report.csv'.format(
                    res_dir, case.upper()),
                '{}/raw/{}/lb-tp-mvapich2/comm-comp-report.csv'.format(
                    res_dir, case.upper()),

                # '{}/raw/{}/mpich3/comm-comp-report.csv'.format(
                #     res_dir, case.upper()),
                # '{}/raw/{}/lb-mpich3/comm-comp-report.csv'.format(
                #     res_dir, case.upper())
            ],
            'lines': {
                'mpi': ['mpich3', 'mvapich2'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '2'],
                'lba': ['500'],
                'b': ['96', '72', '21'],
                't': ['5000'],
                'type': ['total'],
            },
            'outfiles': ['{}/{}-efficiency-DLB-o20.pdf'.format(images_dir, case),
                         '{}/{}-efficiency-DLB-o20.jpg'.format(images_dir, case)]
        },
        # '{} DLB \w TP'.format(case.upper()): {
        #     'files': [
        #         '{}/raw/{}/tp-mvapich2/comm-comp-report.csv'.format(
        #             res_dir, case.upper()),
        #         '{}/raw/{}/lb-tp-mvapich2/comm-comp-report.csv'.format(
        #             res_dir, case.upper()),
        #         # '{}/raw/{}/tp-mpich3/comm-comp-report.csv'.format(
        #         #     res_dir, case.upper()),
        #         # '{}/raw/{}/lb-tp-mpich3/comm-comp-report.csv'.format(
        #         #     res_dir, case.upper())
        #     ],
        #     'lines': {
        #         'mpi': ['mpich3', 'mvapich2'],
        #         'lb': ['interval', 'reportonly'],
        #         'approx': ['0', '2'],
        #         'lba': ['500'],
        #         'b': ['96', '72', '21'],
        #         't': ['5000'],
        #         'type': ['total'],
        #     },
        #     'outfiles': ['{}/{}-DLB-TP.pdf'.format(images_dir, case),
        #                  '{}/{}-DLB-TP.jpg'.format(images_dir, case)]
        # },
    },
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
    'colors': {
        # 'mvapich2': cycle(['xkcd:pastel green', 'xkcd:green', 'xkcd:olive green', 'xkcd:blue green']),
        'mvapich2': cycle([cm.Greens(x) for x in np.linspace(0.2, 0.8, 4)]),
        # 'mpich3-NoLB': cycle(['xkcd:pastel green']),

        # 'mvapich2': cycle(['xkcd:orange', 'xkcd:rust']),
        # 'mvapich2-NoLB': cycle(['xkcd:apricot']),
    },
    'hatches': {
        'LB': 'x',
        'NoLB': '',
    },
    'reference': {
        # 'sps': {'ppb': 4000000, 'b': 72, 'turns': 500, 'w': 1,
        #         'omp': 1, 'time': 1497.8},
        # 'sps': {'ppb': 4000000, 'b': 72, 'turns': 1000, 'w': 1,
        #         'omp': 10, 'time': 415.4},
        'sps': {'ppb': 4000000, 'b': 72, 'turns': 1000, 'w': 1,
                'omp': 20, 'time': 225.85},

        # 'lhc': {'ppb': 2000000, 'b': 96, 'turns': 500, 'w': 1,
        #         'omp': 1, 'time': 681.59},
        # 'lhc': {'ppb': 2000000, 'b': 96, 'turns': 1000, 'w': 1,
        #         'omp': 10, 'time': 177.585},
        'lhc': {'ppb': 2000000, 'b': 96, 'turns': 1000, 'w': 1,
                'omp': 20, 'time': 103.04},

        # 'ps': {'time': 466.085, 'ppb': 8000000, 'b': 21, 'turns': 500},
        # 'ps': {'ppb': 8000000, 'b': 21, 'turns': 500, 'w': 1,
        #        'omp': 1, 'time': 502.88},
        # 'ps': {'ppb': 8000000, 'b': 21, 'turns': 1000, 'w': 1,
        #        'omp': 10, 'time': 142.066},
        'ps': {'ppb': 8000000, 'b': 21, 'turns': 1000, 'w': 1,
               'omp': 20, 'time': 96.0},
    },
    # 'sequence': ['mpich3']

    # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    'x_name': 'n',
    'x_to_keep': [4, 8, 12, 16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    # 'y_err_name': 'std',
    'xlabel': 'Nodes (x20 Cores)',
    'ylabel': 'Efficiency',
    'title': {
        's': '{} DLB'.format(case.upper()),
        'fontsize': 10,
        'y': 0.96,
        'x': 0.5,
        'fontweight': 'bold',
    },
    'figsize': [4.5, 2.5],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 9},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 4, 'handlelength': 1, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.2,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.5,
    },
    'subplots_adjust': {
        'wspace': 0.05, 'hspace': 0.16, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'inout', 'length': 3, 'width': 0.5,
    },
    'ylim': [0, 130],

}

if __name__ == '__main__':
    for title, figconf in config['figures'].items():
        plots_dir = {}
        for file in figconf['files']:
            # print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = get_plots(header, data, figconf['lines'],
                             exclude=figconf.get('exclude', []),
                             prefix=True)
            for key in temp.keys():
                plots_dir['_{}'.format(key)] = temp[key].copy()

        fig = plt.figure(figsize=config['figsize'])

        plt.grid(True, which='major', alpha=0.5)
        plt.grid(False, which='major', axis='x')
        plt.title(**config['title'])
        plt.xlabel(config['xlabel'], labelpad=0, fontsize=config['fontsize'])
        plt.ylabel(config['ylabel'], labelpad=0, fontsize=config['fontsize'])

        pos = 0
        step = 0.1
        width = 1. / (len(plots_dir.keys())+0.5)

        for k in (plots_dir.keys()):
            values = plots_dir[k]
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

            # key = '{}-{}-{}'.format(case, mpiv, lb)

            # label = '{}-{}-{}-{}'.format(mpiv, lb, lba, approx)
            label = '{}-{}'.format(lb, approx)
            color = config['colors']['{}'.format(mpiv)].__next__()
            hatch = config['hatches'][lb]
            # marker = config['markers'][case]
            # ls = config['ls'][case]

            x = get_values(values, header, config['x_name'])
            omp = get_values(values, header, config['omp_name'])
            y = get_values(values, header, config['y_name'])
            parts = get_values(values, header, 'ppb')
            bunches = get_values(values, header, 'b')
            turns = get_values(values, header, 't')

            # This is the throughput
            y = parts * bunches * turns / y

            # Now the reference, 1thread
            yref = config['reference'][case]['time']
            partsref = config['reference'][case]['ppb']
            bunchesref = config['reference'][case]['b']
            turnsref = config['reference'][case]['turns']
            ompref = config['reference'][case]['omp']
            yref = partsref * bunchesref * turnsref / yref

            speedup = y / yref
            x_new = []
            sp_new = []
            for i, xi in enumerate(config['x_to_keep']):
                x_new.append(xi)
                if xi in x:
                    sp_new.append(speedup[list(x).index(xi)])
                else:
                    sp_new.append(0)
            x = np.array(x_new)
            speedup = np.array(sp_new)
            # Efficiency
            speedup = 100 * speedup / (x * omp[0] / ompref)
            x = x * omp[0]

            plt.bar(np.arange(len(x)) + pos, speedup, width=0.8*width,
                    edgecolor='0.', label=label, hatch=hatch,
                    color=color)
            for i, j in zip(np.arange(len(x)) + pos, speedup):
                plt.gca().annotate('{:.0f}'.format(j), xy=(i, j),
                                   **config['annotate'])
            pos += width
        pos += width * step
        plt.ylim(config['ylim'])
        plt.xlim(0-.6*width, len(x)-.6*width)
        plt.xticks(np.arange(len(x)) + 1.5 * width, np.array(x, int)//20)

        plt.legend(**config['legend'])
        plt.gca().tick_params(**config['tick_params'])

        plt.tight_layout()
        plt.subplots_adjust(**config['subplots_adjust'])
        plt.xticks(**config['ticks'])
        plt.yticks(**config['ticks'])
        for file in figconf['outfiles']:
            save_and_crop(fig, file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
