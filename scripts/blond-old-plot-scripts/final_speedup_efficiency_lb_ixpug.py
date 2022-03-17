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

parser.add_argument('-c', '--cases', type=str, nargs='+',
                    choices=['lhc', 'sps', 'ps', 'ex01'],
                    help='The test-case to plot.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')

args = parser.parse_args()

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/redistribute/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

gconfig = {
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
    # 'colors': {
    # 'mvapich2': cycle(['xkcd:pastel green', 'xkcd:green', 'xkcd:olive green', 'xkcd:blue green']),
    # 'mvapich2': cycle([cm.Greens(x) for x in np.linspace(0.2, 0.8, 3)]),
    # 'mpich3-NoLB': cycle(['xkcd:pastel green']),

    # 'mvapich2': cycle(['xkcd:orange', 'xkcd:rust']),
    # 'mvapich2-NoLB': cycle(['xkcd:apricot']),
    # },
    'hatches': {
        'LB': 'x',
        'NoLB': '',
    },
    'hatches': ['', '//', '--', 'xx'],
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
    'ylabel': 'Speedup',
    'ylabel2': 'Efficiency',
    'title': {
                # 's': '{}'.format(case.upper()),
                'fontsize': 10,
                'y': 0.74,
                # 'x': 0.55,
                'fontweight': 'bold',
    },
    'figsize': [5, 3.3],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 10},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 4, 'handlelength': 1.5, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.8,
        'bbox_to_anchor': (0, 1.25)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'inout', 'length': 0, 'width': 0.5,
    },
    'ylim': [0, 8.8],
    'ylim2': [40, 120],
    'yticks': [0, 2, 4, 6, 8],
    'yticks2': [40, 60, 80, 100],
    'outfiles': ['{}/{}-{}-final-speedup-efficiency-o20-ixpug.pdf',
                 '{}/{}-{}-final-speedup-efficiency-o20-ixpug.jpg']
}


lconfig = {
    'figures': {
        # 'NoDLB': {
        #     'files': [
        #         # '{}/raw/{}/mvapich2/comm-comp-report.csv',
        #         '{}/raw/{}/tp-approx0-mvapich2/comm-comp-report.csv',
        #         '{}/raw/{}/approx2-mvapich2/comm-comp-report.csv',
        #         '{}/raw/{}/tp-mvapich2/comm-comp-report.csv',
        #     ],
        #     'lines': {
        #         'mpi': ['mpich3', 'mvapich2'],
        #         'lb': ['interval', 'reportonly'],
        #         'approx': ['0', '2'],
        #         'lba': ['500'],
        #         'b': ['96', '72', '21'],
        #         't': ['5000'],
        #         'type': ['total'],
        #     }
        # },
        'DLB': {
            'files': [
                # '{}/raw/{}/mvapich2/comm-comp-report.csv',
                # '{}/raw/{}/lb-mvapich2/comm-comp-report.csv',
                # '{}/raw/{}/lb-tp-approx0-mvapich2/comm-comp-report.csv',
                # '{}/raw/{}/lb-approx2-mvapich2/comm-comp-report.csv',
                '{}/raw/{}/lb-tp-mvapich2/comm-comp-report.csv',
            ],
            'lines': {
                'mpi': ['mpich3', 'mvapich2'],
                'lb': ['interval', 'reportonly'],
                'approx': ['0', '2'],
                'lba': ['500'],
                'b': ['96', '72', '21'],
                't': ['5000'],
                'type': ['total'],
            }
        },
    },

}

if __name__ == '__main__':
    for title, figconf in lconfig['figures'].items():
        fig, ax_arr = plt.subplots(ncols=1, nrows=len(args.cases),
                                   sharex=True, sharey=True,
                                   figsize=gconfig['figsize'])
        for col, case in enumerate(args.cases):
            ax = ax_arr[col]
            ax2 = ax.twinx()
            plt.sca(ax)
            plots_dir = {}
            for file in figconf['files']:
                file = file.format(res_dir, case.upper())
                # print(file)
                data = np.genfromtxt(file, delimiter='\t', dtype=str)
                header, data = list(data[0]), data[1:]
                temp = get_plots(header, data, figconf['lines'],
                                 exclude=figconf.get('exclude', []),
                                 prefix=True)
                for key in temp.keys():
                    if 'tp-' in file:
                        plots_dir['_{}_tp1'.format(key)] = temp[key].copy()
                    else:
                        plots_dir['_{}_tp0'.format(key)] = temp[key].copy()
            # fig = plt.figure(figsize=config['figsize'])

            # plt.grid(True, which='major', alpha=0.5)
            # plt.grid(False, which='major', axis='x')
            plt.title('{}'.format(case.upper()), **gconfig['title'])
            if col == len(args.cases) - 1:
                plt.xlabel(gconfig['xlabel'], labelpad=3,
                           fontsize=gconfig['fontsize'])
            if col == 1:
                plt.ylabel(gconfig['ylabel'], labelpad=3, color='xkcd:green',
                           fontweight='bold',
                           fontsize=gconfig['fontsize'])
            plt.setp(ax.get_yticklabels(), color="xkcd:green")


            plt.sca(ax2)
            if col == 1:
                plt.ylabel(gconfig['ylabel2'], labelpad=3,
                           fontweight='bold', color='xkcd:blue',
                           fontsize=gconfig['fontsize'])
            plt.setp(ax2.get_yticklabels(), color="xkcd:blue")
            plt.yticks(gconfig['yticks2'], **gconfig['ticks'])
            # if col == len(args.cases) - 1:
            # else:
            #     plt.yticks([])
            plt.ylim(gconfig['ylim2'])
            plt.sca(ax)

            pos = 0
            step = 0.1
            width = 1. / (2*len(plots_dir.keys())+0.4)

            # colors = [cm.Greens(x) for x in np.linspace(0.2, 0.8, len(plots_dir))]
            colors1 = [cm.Greens(x)
                       for x in np.linspace(0.5, 0.8, len(plots_dir))]
            colors2 = [cm.Blues(x)
                       for x in np.linspace(0.5, 0.8, len(plots_dir))]
            for idx, k in enumerate(plots_dir.keys()):
                values = plots_dir[k]
                mpiv = k.split('_mpi')[1].split('_')[0]
                lb = k.split('lb')[1].split('_')[0]
                lba = k.split('lba')[1].split('_')[0]
                approx = k.split('approx')[1].split('_')[0]
                tp = k.split('tp')[1].split('_')[0]
                if lb == 'interval':
                    lb = 'LB'
                elif lb == 'reportonly':
                    lb = 'NoLB'
                if tp == '1':
                    tp = 'TP'
                elif tp == '0':
                    tp = 'NoTP'
                if approx == '2':
                    approx = 'AC'
                else:
                    approx = 'NoAC'
                # key = '{}-{}-{}'.format(case, mpiv, lb)

                label = '{}-{}-{}'.format(lb, tp, approx)
                # label = '{}-{}'.format(tp, approx)
                # color = gconfig['colors']['{}'.format(mpiv)].__next__()
                # hatch = gconfig['hatches'][lb]
                # marker = config['markers'][case]
                # ls = config['ls'][case]

                x = get_values(values, header, gconfig['x_name'])
                omp = get_values(values, header, gconfig['omp_name'])
                y = get_values(values, header, gconfig['y_name'])
                parts = get_values(values, header, 'ppb')
                bunches = get_values(values, header, 'b')
                turns = get_values(values, header, 't')

                # This is the throughput
                y = parts * bunches * turns / y

                # Now the reference, 1thread
                yref = gconfig['reference'][case]['time']
                partsref = gconfig['reference'][case]['ppb']
                bunchesref = gconfig['reference'][case]['b']
                turnsref = gconfig['reference'][case]['turns']
                yref = partsref * bunchesref * turnsref / yref
                ompref = gconfig['reference'][case]['omp']

                speedup = y / yref
                x_new = []
                sp_new = []
                for i, xi in enumerate(gconfig['x_to_keep']):
                    x_new.append(xi)
                    if xi in x:
                        sp_new.append(speedup[list(x).index(xi)])
                    else:
                        sp_new.append(0)
                x = np.array(x_new)
                speedup = np.array(sp_new)
                efficiency = 100 * speedup / (x * omp[0] / ompref)
                x = x * omp[0]

                # efficiency = 100 * speedup / x
                plt.bar(np.arange(len(x)) + pos, speedup, width=0.9*width,
                        edgecolor='0.', label=label, hatch=gconfig['hatches'][idx],
                        color=colors1[idx])
                ax2.bar(np.arange(len(x)) + pos + width, efficiency, width=0.9*width,
                        edgecolor='0.', label=label, hatch=gconfig['hatches'][idx],
                        color=colors2[idx])
                if idx != 1:
                    for i, s, e in zip(np.arange(len(x)) + pos, speedup, efficiency):
                        ax.annotate('{:.1f}'.format(s), xy=(i, s),
                                    **gconfig['annotate'])
                        ax2.annotate('{:.0f}'.format(e), xy=(i+width, e),
                                     **gconfig['annotate'])
                pos += width
            pos += width * step
            plt.ylim(gconfig['ylim'])
            plt.xlim(0-.8*width, len(x)-.7*width)
            plt.xticks(np.arange(len(x)) + width/2, np.array(x, int)//20)

            if col == 0:
                handles = []
                handles.append(mpatches.Patch(label='Speedup', edgecolor='black',
                                              facecolor=colors1[0]))
                handles.append(mpatches.Patch(label='Efficiency', edgecolor='black',
                                              facecolor=colors2[0]))

                plt.legend(handles=handles, **gconfig['legend'])

            # ax.tick_params(**gconfig['tick_params'])
            # ax.tick_params(axis='y', color=colors1[0])

            # ax2.tick_params(**gconfig['tick_params'])
            # ax2.tick_params(axis='y', color=colors2[0])
            plt.xticks(**gconfig['ticks'])
            plt.yticks(gconfig['yticks'], **gconfig['ticks'])

        plt.tight_layout()
        plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir, title, '-'.join(args.cases))
            save_and_crop(fig, file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
