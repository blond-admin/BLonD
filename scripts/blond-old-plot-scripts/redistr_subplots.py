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
    ('LHC', 'lb-mpich3'),
    ('LHC', 'mpich3'),
    # ('LHC', 'lb-mpich3-approx2'),
    # ('LHC', 'lb-openmpi3'),
    # ('LHC', 'lb-openmpi3-approx2'),
    # ('LHC', 'lb-mvapich2'),
    # ('LHC', 'lb-mvapich2-approx2'),
    # ('LHC', 'dynamic-lb-mpich3'),
    # ('LHC', 'dynamic-lb-mpich3-approx2'),
    # # ('LHC', 'dynamic-lb-openmpi3'),
    # # ('LHC', 'dynamic-lb-openmpi3-approx2'),
    # # ('LHC', 'mpich3'),
    # # ('LHC', 'mvapich2'),
    # # ('LHC', 'openmpi3'),
    # ('LHC', 'dynamic-lb-mvapich2'),
    # ('LHC', 'dynamic-lb-mvapich2-approx2'),
    # ('SPS', 'lb-mpich3'),
    # ('SPS', 'lb-mpich3-approx2'),
    # # ('SPS', 'lb-openmpi3'),
    # # ('SPS', 'lb-openmpi3-approx2'),
    # ('SPS', 'lb-mvapich2'),
    # ('SPS', 'lb-mvapich2-approx2'),
    # ('SPS', 'dynamic-lb-mpich3'),
    # ('SPS', 'dynamic-lb-mpich3-approx2'),
    # # ('SPS', 'dynamic-lb-openmpi3'),
    # # ('SPS', 'dynamic-lb-openmpi3-approx2'),
    # ('SPS', 'dynamic-lb-mvapich2'),
    # ('SPS', 'dynamic-lb-mvapich2-approx2'),
    # ('PS', 'lb-mpich3'),
    # ('PS', 'lb-mpich3-approx2'),
    # # ('PS', 'lb-openmpi3'),
    # # ('PS', 'lb-openmpi3-approx2'),
    # ('PS', 'lb-mvapich2'),
    # ('PS', 'lb-mvapich2-approx2'),
    # ('PS', 'dynamic-lb-mpich3'),
    # ('PS', 'dynamic-lb-mpich3-approx2'),
    # # ('PS', 'dynamic-lb-openmpi3'),
    # # ('PS', 'dynamic-lb-openmpi3-approx2'),
    # ('PS', 'dynamic-lb-mvapich2'),
    # ('PS', 'dynamic-lb-mvapich2-approx2'),
    # ('PS', 'mpich3'),
    # ('PS', 'mvapich2'),
    # ('PS', 'openmpi3'),
    # ('EX01', 'lb-mpich3'),
    # ('EX01', 'lb-mpich3-approx2'),
    # ('EX01', 'mpich3'),
]
conf = {
    'files': {
        '{}/local_raw/{}/{}/particles-report.csv': {
            'dic_rows': ['n'],
            'dic_cols': ['turn_num', 'parts_avg', 'parts_min',
                         'parts_max', 'tcomp_avg', 'tcomp_min',
                         'tcomp_max', 'tcomm_avg', 'tcomm_min',
                         'tcomm_max', 'tconst_avg', 'tconst_min',
                         'tconst_max', 'tpp_avg',
                         'tpp_min', 'tpp_max'],
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
            'y_name': ['tconst_avg'],
            'y_min_name': ['tconst_min'],
            'y_max_name': ['tconst_max']
         },
        {'title': 'tcomp',
            'x_name': 'turn_num',
            'y_name': ['tcomp_avg'],
            'y_min_name': ['tcomp_min'],
            'y_max_name': ['tcomp_max']
         },
        {'title': 'tcomm',
            'x_name': 'turn_num',
            'y_name': ['tcomm_avg'],
            'y_min_name': ['tcomm_min'],
            'y_max_name': ['tcomm_max']
         },
        {'title': 'ttotal',
            'x_name': 'turn_num',
            'y_name': ['tconst_avg', 'tcomm_avg', 'tcomp_avg'],
            'y_min_name': ['tconst_min', 'tcomm_min', 'tcomp_min'],
            'y_max_name': ['tconst_max', 'tcomm_max', 'tcomp_max']
         },
        {'title': 'parts',
            'x_name': 'turn_num',
            'y_name': ['parts_avg'],
            'y_min_name': ['parts_min'],
            'y_max_name': ['parts_max']
         },
        {'title': 'tpp',
            'x_name': 'turn_num',
            'y_name': ['tpp_avg'],
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
    'outfiles': ['{}/{}-subplots.pdf'],
    'points': 10,
}

if __name__ == '__main__':
    for case_tup in cases:
        case = case_tup[0] + '-' + case_tup[1]
        plots_dir = {}
        for file, fconfig in conf['files'].items():
            file = file.format(res_dir, case_tup[0], case_tup[1])
            print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = dictify(
                header, data, fconfig['dic_rows'], fconfig['dic_cols'])
            plots_dir.update(temp)

        fig, ax_arr = plt.subplots(**conf['subplots_args'], sharex=True)
        fig.suptitle(case.upper())
        i = 0
        for pltconf in conf['subplots']:
            ax = ax_arr[i//conf['subplots_args']['ncols'],
                        i % conf['subplots_args']['ncols']]
            plt.sca(ax)
            plt.title(pltconf['title'], fontsize=8)
            step = 1
            pos = 0
            width = step / (len(plots_dir.keys()) + 1)
            for nw, data in plots_dir.items():
                x = np.array(data[pltconf['x_name']], float)
                y = np.sum([np.array(data[y_name], float)
                            for y_name in pltconf['y_name']], axis=0)
                ymin = np.sum([np.array(data[y_name], float)
                               for y_name in pltconf['y_min_name']], axis=0)
                ymax = np.sum([np.array(data[y_name], float)
                               for y_name in pltconf['y_max_name']], axis=0)
                if x[0] == 0:
                    x, y, ymin, ymax = x[1:], y[1:], ymin[1:], ymax[1:]
                idx = np.linspace(0, len(x)-1, conf['points'], dtype=int)
                if pltconf['title'] in ['tconst', 'tcomm',
                                        'tcomp', 'tsync', 'ttotal', 'tpp']:
                    turndiff = np.diff(np.insert(x, 0, 0))
                    y /= turndiff
                    ymin /= turndiff * y[0]
                    ymax /= turndiff * y[0]
                    y /= y[0]
                    # ymin = np.cumsum(ymin) / x / (y[0]/x[0])
                    # ymax = np.cumsum(ymax) / x / (y[0]/x[0])
                    # y = np.cumsum(y) / x / (y[0]/x[0])
                    plt.axhline(1, color='k', ls='dotted', lw=1, alpha=0.5)
                    if np.max(ymax) > 2.:
                        plt.ylim(top=2.)
                x, y, ymin, ymax = x[idx], y[idx], ymin[idx], ymax[idx]

                plt.bar(np.arange(len(x)) + pos, y, width=width,
                        label='{}'.format(nw), lw=1, edgecolor='0',
                        color=conf['colors'][nw],
                        yerr=[y-ymin, ymax-y], error_kw={'capsize': 2, 'elinewidth': 1})
                pos += 1.05*width
                # plt.errorbar(x + displ, y, yerr=[ymin, ymax], label='{}'.format(nw),
                #              lw=1, capsize=1, color=conf['colors'][nw])
            plt.xticks(np.arange(len(x))+pos/2, np.array(x, int))
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
