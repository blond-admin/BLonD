#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import matplotlib.patches as mpatches
import sys
from plot.plotting_utilities import *
from cycler import cycle

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/redistribute/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

cases = [
    'PS-lb-mpich3-test',
    # 'LHC-lb-mpich3',
    # 'LHC-lb-mpich3-approx2',
    # 'LHC-lb-openmpi3',
    # 'LHC-lb-openmpi3-approx2',
    # 'LHC-lb-mvapich2',
    # 'LHC-lb-mvapich2-approx2',
    # 'SPS-lb-mpich3',
    # 'SPS-lb-mpich3-approx2',
    # 'SPS-lb-openmpi3',
    # 'SPS-lb-openmpi3-approx2',
    # 'SPS-lb-mvapich2',
    # 'SPS-lb-mvapich2-approx2',
    # 'PS-lb-mpich3',
    # 'PS-lb-mpich3-approx2',
    # 'PS-lb-openmpi3',
    # 'PS-lb-openmpi3-approx2',
    # 'PS-lb-mvapich2',
    # 'PS-lb-mvapich2-approx2',
    # 'LHC-mpich3',
    # 'LHC-mvapich2',
    # 'LHC-openmpi3',
    # 'PS-mpich3',
    # 'PS-mvapich2',
    # 'PS-openmpi3',
    # 'EX01-lb-mpich3',
    # 'EX01-lb-mpich3-approx2',
    # 'EX01-mpich3',
]
conf = {
    'files': {
        '{}/raw/{}/particles-distribution-report.csv': {
            'dic_rows': ['n', 'turn_num'],
            'dic_cols': ['turn_num', 'parts', 'tcomp', 'tcomm', 'tconst', 'tpp'],
        },
    },
    'subplots_args': {
        'nrows': 3,
        'ncols': 2,
        'figsize': (14, 7)
    },
    'subplots': [
        {'title': 'tconst',
            'x_name': 'turn_num',
            'y_name': ['tconst'],
            'y_min_name': ['tconst_min'],
            'y_max_name': ['tconst_max']
         },
        {'title': 'tcomp',
            'x_name': 'turn_num',
            'y_name': ['tcomp'],
            'y_min_name': ['tcomp_min'],
            'y_max_name': ['tcomp_max']
         },
        {'title': 'tcomm',
            'x_name': 'turn_num',
            'y_name': ['tcomm'],
            'y_min_name': ['tcomm_min'],
            'y_max_name': ['tcomm_max']
         },
        {'title': 'ttotal',
            'x_name': 'turn_num',
            'y_name': ['tconst', 'tcomm', 'tcomp'],
            'y_min_name': ['tconst_min', 'tcomm_min', 'tcomp_min'],
            'y_max_name': ['tconst_max', 'tcomm_max', 'tcomp_max']
         },
        {'title': 'parts',
            'x_name': 'turn_num',
            'y_name': ['parts'],
            'y_min_name': ['parts_min'],
            'y_max_name': ['parts_max']
         },
        {'title': 'tpp',
            'x_name': 'turn_num',
            'y_name': ['tpp'],
            'y_min_name': ['tpp_min'],
            'y_max_name': ['tpp_max']
         }
    ],
    'colors': {'2': 'xkcd:light blue',
               '4': 'xkcd:light orange',
               '8': 'xkcd:light green',
               '12': 'xkcd:light red',
               '16': 'xkcd:light purple'
               },
    'legend': {
        'loc': 'upper left', 'ncol': 5, 'handlelength': 1, 'fancybox': True,
        'framealpha': 0., 'fontsize': 9, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.5,
    },
    'subplots_adjust': {
        'wspace': 0.05, 'hspace': 0.16, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 1, 'bottom': 1, 'left': 1,
        'direction': 'inout', 'length': 3, 'width': 0.5,
    },
    'outfiles': ['{}/{}-violin-subplots.pdf'],
    'points': 10,
}

if __name__ == '__main__':
    for case in cases:
        plots_dir = {}
        for file, fconfig in conf['files'].items():
            file = file.format(res_dir, case)
            print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = dictify(header, data, fconfig['dic_rows'],
                           fconfig['dic_cols'])
            plots_dir.update(temp)

        fig, ax_arr = plt.subplots(**conf['subplots_args'], sharex=True)
        fig.suptitle(case.upper())
        i = 0
        turns = set([float(k.split('-')[2]) for k in plots_dir.keys()])
        turns = sorted(turns)
        idx = np.linspace(0, len(turns)-1, conf['points'], dtype=int)
        turns = np.array(turns)[idx]
        for pltconf in conf['subplots']:
            ax = ax_arr[i//conf['subplots_args']['ncols'],
                        i % conf['subplots_args']['ncols']]
            plt.sca(ax)
            plt.title(pltconf['title'], fontsize=8)
            step = 1
            pos = 0
            width = step / (len(plots_dir.keys()) + 1)
            for key, data in plots_dir.items():
                nw = key.split('-')[1]
                turn_num = key.split('-')[2]
                if float(turn_num) not in turns:
                    continue
                # x = np.array(data[pltconf['x_name']], float)
                # idx = np.linspace(0, len(x)-1, conf['points'], dtype=int)
                # x = x[idx]
                y = np.sum([np.array(data[y_name], float)
                            for y_name in pltconf['y_name']], axis=0)
                # if pltconf['title'] in ['tconst', 'tcomm', 'tcomp', 'ttotal', 'tpp']:
                #     y = np.cumsum(y) / x / (y[0]/x[0])
                #     plt.axhline(1, color='k', ls='dotted', lw=1, alpha=0.5)
                #     if np.max(ymax) > 2.:
                #         plt.ylim(top=2.)

                plt.violinplot([y], [pos])
                
                pos += width

                # plt.bar(np.arange(len(x)) + pos, y, width=width,
                #         label='{}'.format(nw), lw=1, edgecolor='0',
                #         color=conf['colors'][nw],
                #         yerr=[y-ymin, ymax-y], error_kw={'capsize': 2, 'elinewidth': 1})
                pos += 1.05*width
                # plt.errorbar(x + displ, y, yerr=[ymin, ymax], label='{}'.format(nw),
                #              lw=1, capsize=1, color=conf['colors'][nw])
            # plt.xticks(np.arange(len(x))+pos/2, np.array(x, int))
            ax.tick_params(**conf['tick_params'])
            plt.legend(**conf['legend'])
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            i += 1

        plt.subplots_adjust(**conf['subplots_adjust'])
        for outfile in conf['outfiles']:
            outfile = outfile.format(images_dir, case)
            save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
