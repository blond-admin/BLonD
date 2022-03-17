import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import matplotlib.ticker
import sys
from plot.plotting_utilities import *

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
# images_dir = res_dir + 'plots/redistribute/'
images_dir = res_dir + '/blond-meeting/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

case = 'ps'

config = {

    'files': {

        '{}/raw/PS/mvapich2/comm-comp-report.csv'.format(res_dir): {
        # '{}/raw/PS/b21-t10k-mvapich2/comm-comp-report.csv'.format(res_dir): {
            'key': 'ps-mvapich2',
            'lines': {
                'omp': ['10'],
                'type': ['total']
            }
        },

        '{}/blond-meeting/PS/lb-mvapich2/comm-comp-report.csv'.format(res_dir): {
            'key': 'ps-lbmvapich2',
            'lines': {
                'omp': ['10'],
                'type': ['total']
            }
        },
        # '{}/raw/SPS/b72-4MPPB-t10k-mvapich2/comm-comp-report.csv'.format(res_dir): {
        '{}/blond-meeting/SPS/mvapich2/comm-comp-report.csv'.format(res_dir): {
            'key': 'sps-mvapich2',
            'lines': {
                'omp': ['10'],
                'type': ['total']
            }
        },

        '{}/blond-meeting/SPS/lb-mvapich2/comm-comp-report.csv'.format(res_dir): {
            'key': 'sps-lbmvapich2',
            'lines': {
                'omp': ['10'],
                'type': ['total']
            }
        },
        '{}/raw/LHC/mvapich2/comm-comp-report.csv'.format(res_dir): {
            'key': 'lhc-mvapich2',
            'lines': {
                'omp': ['10'],
                'type': ['total']
            }
        },

        '{}/blond-meeting/LHC/lb-mvapich2/comm-comp-report.csv'.format(res_dir): {
            'key': 'lhc-lbmvapich2',
            'lines': {
                'omp': ['10'],
                'type': ['total']
            }
        },    

    },
    'labels': {

        # '{}-mpich3'.format(case): '{}-mpich3'.format(case),
        # '{}-lbmpich3'.format(case): '{}-lb-mpich3'.format(case),
        # '{}-mpich3apprx'.format(case): '{}-mpich3-approx2'.format(case),
        # '{}-dynlbmpich3'.format(case): '{}-dynamic-lb-mpich3'.format(case),
        # '{}-lbmpich3apprx'.format(case): '{}-lb-mpich3-approx2'.format(case),
        # '{}-dynlbmpich3apprx'.format(case): '{}-dynamic-lb-mpich3-approx2'.format(case),

        'ps-mvapich2': 'PS',
        'ps-lbmvapich2': 'PS-TP',

        'sps-mvapich2': 'SPS',
        'sps-lbmvapich2': 'SPS-TP',

        'lhc-mvapich2': 'LHC',
        'lhc-lbmvapich2': 'LHC-TP',

        # '{}-mvapich2apprx'.format(case): '{}-mvapich2-approx2'.format(case),
        # '{}-dynlbmvapich2'.format(case): '{}-dynamic-lb-mvapich2'.format(case),
        # '{}-lbmvapich2apprx'.format(case): '{}-lb-mvapich2-approx2'.format(case),
        # '{}-dynlbmvapich2apprx'.format(case): '{}-dynamic-lb-mvapich2-approx2'.format(case),

        # '{}-openmpi3'.format(case): '{}-openmpi3'.format(case),
        # '{}-lbopenmpi3'.format(case): '{}-lb-openmpi3'.format(case),
        # '{}-openmpi3apprx'.format(case): '{}-openmpi3-approx2'.format(case),
        # '{}-dynlbopenmpi3'.format(case): '{}-dynamic-lb-openmpi3'.format(case),
        # '{}-lbopenmpi3apprx'.format(case): '{}-lb-openmpi3-approx2'.format(case),
        # '{}-dynlbopenmpi3apprx'.format(case): '{}-dynamic-lb-openmpi3-approx2'.format(case),

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
        # 'mpich3': 'xkcd:light yellow',
        # 'lbmpich3': 'xkcd:yellow',
        # # 'mpich3apprx': 'xkcd:light green',
        # 'lbmpich3apprx': 'xkcd:green',

        'ps-mvapich2': 'xkcd:light orange',
        'ps-lbmvapich2': 'xkcd:orange',

        'sps-mvapich2': 'xkcd:light blue',
        'sps-lbmvapich2': 'xkcd:blue',

        'lhc-mvapich2': 'xkcd:light green',
        'lhc-lbmvapich2': 'xkcd:green',


        # 'mvapich2apprx': 'xkcd:light red',
        # 'lbmvapich2apprx': 'xkcd:red',

        # 'openmpi3': 'xkcd:light pink',
        # 'lbopenmpi3': 'xkcd:pink',
        # 'openmpi3apprx': 'xkcd:light purple',
        # 'lbopenmpi3apprx': 'xkcd:purple',

        # 'dynlbmpich3': 'xkcd:light pink',
        # 'dynlbmpich3apprx': 'xkcd:pink',

        # 'dynlbmvapich2': 'xkcd:light purple',
        # 'dynlbmvapich2apprx': 'xkcd:purple',


    },
    'hatches': {
        'mpich3': 'x',
        'openmpi3': '-',
        'mvapich2': 'o',
        # 'ex01': '\\',
        # 'lhc': '/',
        # 'sps': 'o',
        # 'ps': 'x',
    },
    'reference': {
        'ex01': {'time': 21.4, 'ppb': 1000000, 'turns': 2000},
        'sps': {'time': 430., 'ppb': 4000000, 'turns': 100},
        'lhc': {'time': 2120., 'ppb': 2000000, 'turns': 1000},
        'ps': {'time': 1623.7, 'ppb': 4000000, 'turns': 2000},
    },

    # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    'x_name': 'n',
    'x_to_keep': [4, 8, 12, 16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    # 'y_err_name': 'std',
    'xlabel': 'Cores (x10)',
    'ylabel': 'Speedup',
    'title': 'Task Parallelism',
    'figsize': (5, 3.5),
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 3, 'handlelength': 1, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0, 'columnspacing': 0.5,
    },
    'subplots_adjust': {
        'wspace': 0.05, 'hspace': 0.16, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 1, 'bottom': 1, 'left': 1,
        'direction': 'inout', 'length': 3, 'width': 0.5,
    },
    'image_name': '{}/task-parallel.pdf'.format(images_dir),

}

if __name__ == '__main__':
    plots_dir = {}
    for file in config['files'].keys():
        # print(file)
        data = np.genfromtxt(file, delimiter='\t', dtype=str)
        header, data = list(data[0]), data[1:]
        temp = get_plots(header, data, config['files'][file]['lines'],
                         exclude=config['files'][file].get('exclude', []))
        temp[config['files'][file]['key']] = temp['10-total']
        del temp['10-total']
        plots_dir.update(temp)

    fig = plt.figure(figsize=config['figsize'])

    plt.grid(True, which='major', alpha=0.5)
    plt.grid(False, which='major', axis='x')
    plt.title(config['title'])
    plt.xlabel(config['xlabel'], fontsize=config['fontsize'])
    plt.ylabel(config['ylabel'], fontsize=config['fontsize'])

    pos = 0
    step = 0.1
    width = 1. / (len(plots_dir.keys())+1)
    for case in ['lhc', 'sps', 'ps', 'ex01']:
        for mpiv in ['mpich3', 'lbmpich3',
                     'mpich3apprx', 'lbmpich3apprx',
                     'dynlbmpich3', 'dynlbmpich3apprx',
                     'mvapich2', 'lbmvapich2',
                     'mvapich2apprx', 'lbmvapich2apprx',
                     'dynlbmvapich2', 'dynlbmvapich2apprx',
                     'openmpi3', 'lbopenmpi3',
                     'openmpi3apprx', 'lbopenmpi3apprx']:

            key = '{}-{}'.format(case, mpiv)
            if key not in plots_dir:
                continue
            if 'mpich3' in mpiv:
                version = 'mpich3'
            elif 'mvapich2' in mpiv:
                version = 'mvapich2'
            elif 'openmpi3' in mpiv:
                version = 'openmpi3'
            values = plots_dir[key]

            label = config['labels'][key]

            x = get_values(values, header, config['x_name'])
            omp = get_values(values, header, config['omp_name'])
            y = get_values(values, header, config['y_name'])
            parts = get_values(values, header, 'ppb')
            turns = get_values(values, header, 'turns')

            # y = np.array(values[:, header.index(config['y_name'])], float)
            # parts = np.array(values[:, header.index('ppb')], float)
            # turns = np.array(values[:, header.index('turns')], float)
            # This is the throughput
            y = parts * turns / y

            # Now the reference, 1thread
            yref = config['reference'][case]['time']
            partsref = config['reference'][case]['ppb']
            turnsref = config['reference'][case]['turns']
            yref = partsref * turnsref / yref

            speedup = y / yref

            if len(config['x_to_keep']) < len(x):
                x_new = []
                speedup_new = []
                omp_new = []
                for i in range(len(x)):
                    if x[i] in config['x_to_keep']:
                        x_new.append(x[i])
                        speedup_new.append(speedup[i])
                        omp_new.append(omp[i])
                x = np.array(x_new)
                speedup = np.array(speedup_new)
                omp = np.array(omp_new)

            x = x * omp

            # efficiency = 100 * speedup / x
            plt.bar(np.arange(len(x)) + pos, speedup, width=width,
                    color=config['colors'][key],
                    label=label,
                    edgecolor='0',
                    # alpha=0.8,
                    # hatch=config['hatches'][version]
                    )


            pos += width
        pos += 1.5 * width * step

    plt.xticks(np.arange(len(x)) + pos/2.2, np.array(x//10, int))

    # handles = []
    # for k, v in config['colors'].items():
    #     patch = mpatches.Patch(label=k, edgecolor='black', facecolor=v,
    #                            linewidth=.5, alpha=0.9)
    #     handles.append(patch)

    # for k, v in config['hatches'].items():
    #     patch = mpatches.Patch(label=k, edgecolor='black',
    #                            facecolor='0.8', hatch=v, linewidth=.5,)
    #     handles.append(patch)

    # plt.legend(handles=handles, **config['legend'])
    plt.legend(**config['legend'])
    plt.gca().tick_params(**config['tick_params'])

    plt.subplots_adjust(**config['subplots_adjust'])
    plt.xticks(fontsize=config['fontsize'])
    plt.yticks(fontsize=config['fontsize'])
    plt.tight_layout()
    save_and_crop(fig, config['image_name'], dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
