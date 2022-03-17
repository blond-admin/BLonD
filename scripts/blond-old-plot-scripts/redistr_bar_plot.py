#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import matplotlib.patches as mpatches
import sys
from plot.plotting_utilities import *
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/redistribute/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)


plots_config = {

    'plot4': {
        'files': {
            # res_dir+'raw/EX01-mpich3/avg-report.csv': {
            #         'key': 'ex01-mpich3',
            #         'lines': {
            #             # 'omp': ['10'],'
            #             'function': None}
            # },
            # res_dir+'raw/EX01-lb-mpich3/avg-report.csv': {
            #         'key': 'ex01-lbmpich3',
            #         'lines': {
            #             # 'omp': ['10'],'
            #             'function': None}
            # },

            # res_dir+'raw/EX01-mpich3-approx2/avg-report.csv': {
            #         'key': 'ex01-mpich3apprx',
            #         'lines': {
            #             # 'omp': ['10'],'
            #             'function': None}
            # },
            # res_dir+'raw/EX01-lb-mpich3-approx2/avg-report.csv': {
            #         'key': 'ex01-lbmpich3apprx',
            #         'lines': {
            #             # 'omp': ['10'],'
            #             'function': None}
            # },
            res_dir+'raw/LHC-96B-2MPPB-t10k-mpich3/avg-report.csv': {
                    'key': 'lhc-mpich3',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
            res_dir+'raw/LHC-lb-mpich3/avg-report.csv': {
                    'key': 'lhc-lbmpich3',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },

            res_dir+'raw/LHC-approx2-mpich3/avg-report.csv': {
                    'key': 'lhc-mpich3apprx',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
            res_dir+'raw/LHC-lb-mpich3-approx2/avg-report.csv': {
                    'key': 'lhc-lbmpich3apprx',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },

            res_dir+'raw/LHC-96B-2MPPB-t10k-mvapich2/avg-report.csv': {
                    'key': 'lhc-mvapich2',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
            res_dir+'raw/LHC-lb-mvapich2/avg-report.csv': {
                    'key': 'lhc-lbmvapich2',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },

            res_dir+'raw/LHC-approx2-mvapich2/avg-report.csv': {
                    'key': 'lhc-mvapich2apprx',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
            res_dir+'raw/LHC-lb-mvapich2-approx2/avg-report.csv': {
                    'key': 'lhc-lbmvapich2apprx',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
            res_dir+'raw/LHC-96B-2MPPB-t10k-openmpi3/avg-report.csv': {
                    'key': 'lhc-openmpi3',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
            res_dir+'raw/LHC-lb-openmpi3/avg-report.csv': {
                    'key': 'lhc-lbopenmpi3',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },

            res_dir+'raw/LHC-approx2-openmpi3/avg-report.csv': {
                    'key': 'lhc-openmpi3apprx',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
            res_dir+'raw/LHC-lb-openmpi3-approx2/avg-report.csv': {
                    'key': 'lhc-lbopenmpi3apprx',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
        },
        'colors': {
            # 'comm': 'tab:green',
            # 'comp': 'tab:blue',
            # 'serial': 'tab:orange',
            # 'other': 'tab:purple'
            # 'comp': '0.95',
            # 'serial': '0.85',
            'comm': '0.75',
            'redistr': '0.5',
        },
        'hatches': {
            'mpich3': 'x',
            'openmpi3': '-',
            'mvapich2': 'o',
            # 'ex01': '\\',
            # 'lhc': '/',
            # 'sps': 'o',
            # 'ps': 'x',

            # 'mpich3': '/',
            # 'openmpi3': 'o',
            # 'mvapich2': 'x',
        },
        'edgecolor': {
            'mpich3': 'xkcd:light yellow',
            'lbmpich3': 'xkcd:yellow',
            'mpich3apprx': 'xkcd:light green',
            'lbmpich3apprx': 'xkcd:green',

            'mvapich2': 'xkcd:light orange',
            'lbmvapich2': 'xkcd:orange',
            'mvapich2apprx': 'xkcd:light red',
            'lbmvapich2apprx': 'xkcd:red',

            'openmpi3': 'xkcd:light pink',
            'lbopenmpi3': 'xkcd:pink',
            'openmpi3apprx': 'xkcd:light purple',
            'lbopenmpi3apprx': 'xkcd:purple',

        },

        'x_name': 'n',
        'x_to_keep': [2, 4, 8, 12, 16],

        'omp_name': 'omp',
        'y_name': 'global_percentage',
        'y_err_name': 'std',
        'xlabel': 'Cores (x10)',
        'ylabel': 'Percent',
        'title': 'LHC Load-Balance Run-Time Breakdown',
        'ylim': [0, 45],
        'figsize': (8, 6),
        'fontsize': 8,
        'legend': {
            'loc': 'upper left', 'ncol': 3, 'handlelength': 1, 'fancybox': True,
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
        'image_name': images_dir + 'lhc-lb-time-breakdown.pdf'

    },


}

if __name__ == '__main__':
    for plot_key, config in plots_config.items():
        plots_dir = {}
        for file in config['files'].keys():
            print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = get_plots(header, data, config['files'][file]['lines'],
                             exclude=config['files'][file].get('exclude', []))
            for k, v in temp.items():
                plots_dir['{}-{}'.format(config['files'][file]['key'], k)] = v

        final_dir = {}

        for key in plots_dir.keys():
            tc = key.split('-')[0]
            mpi = key.split('-')[1]
            func = key.split('-')[2]
            if func == 'total_time':
                continue
            if func in ['Other', 'other']:
                func = 'serial'
            if tc not in final_dir:
                final_dir[tc] = {}
            if mpi not in final_dir[tc]:
                final_dir[tc][mpi] = {}
            # if 'hostsync' in func:
            #     func = 'sync'
            if 'redistribute' in func:
                func = 'redistr'
            # elif 'sync' in func:
            #     func = 'sync'
            else:
                func = func.split(':')[0]

            if func in final_dir[tc][mpi]:
                final_dir[tc][mpi][func]['y'] += get_values(
                    plots_dir[key], header, config['y_name'])
            else:
                final_dir[tc][mpi][func] = {}
                final_dir[tc][mpi][func]['y'] = get_values(
                    plots_dir[key], header, config['y_name'])
                final_dir[tc][mpi][func]['x'] = get_values(
                    plots_dir[key], header, config['x_name'])

        # print(final_dir)

        fig = plt.figure(figsize=config['figsize'])
        plt.grid(False, axis='y')
        plt.title(config['title'])
        plt.xlabel(config['xlabel'], fontsize=config['fontsize'])
        plt.ylabel(config['ylabel'], fontsize=config['fontsize'])
        if 'ylim' in config:
            plt.ylim(config['ylim'])

        pos = 0.

        for tc in final_dir.keys():
            width = 1. / (len(final_dir[tc].keys())+1)
            intra_step = width
            inter_step = width/2.5
            for mpiv in final_dir[tc].keys():
                if 'mpich3' in mpiv:
                    version = 'mpich3'
                elif 'mvapich2' in mpiv:
                    version = 'mvapich2'
                elif 'openmpi3' in mpiv:
                    version = 'openmpi3'
                # funcs = list(final_dir[tc][mpiv].keys())
                # funcs.sort()
                bottom = None
                for f in ['comm', 'redistr']:
                    if f not in final_dir[tc][mpiv]:
                        continue
                    x = final_dir[tc][mpiv][f]['x']
                    y = final_dir[tc][mpiv][f]['y']
                    if len(config['x_to_keep']) < len(x):
                        x_new = []
                        y_new = []
                        for i in range(len(x)):
                            if x[i] in config['x_to_keep']:
                                x_new.append(x[i])
                                y_new.append(y[i])
                        x = np.array(x_new)
                        y = np.array(y_new)
                    if bottom is None:
                        bottom = np.zeros(len(y))
                    plt.bar(np.arange(len(x)) + pos, y,
                            width=width,
                            bottom=bottom,
                            capsize=1, linewidth=1.,
                            edgecolor=config['edgecolor'][mpiv],
                            color=config['colors'][f],
                            hatch=config['hatches'][version])
                    # label='')
                    bottom += y
                pos += intra_step
            pos += inter_step
        plt.xticks(np.arange(len(x))+4*width, x.astype(int))

        handles = []
        for k, v in config['edgecolor'].items():
            patch = mpatches.Patch(label=k, edgecolor='black', facecolor=v,
                                   linewidth=.5,)
            handles.append(patch)

        for k, v in config['colors'].items():
            patch = mpatches.Patch(label=k, edgecolor='black', facecolor=v,
                                   linewidth=.5,)
            handles.append(patch)

        for k, v in config['hatches'].items():
            patch = mpatches.Patch(label=k, edgecolor='black',
                                   facecolor='0.9', hatch=v, linewidth=.5,)
            handles.append(patch)

        # plt.legend(loc='best', fancybox=True, fontsize=9.5,
        # plt.legend(handles=handles, loc='lower left', fancybox=True,
        #            fontsize=9, ncol=2, columnspacing=1,
        #            labelspacing=0.1, borderpad=0.5, framealpha=0.7,
        #            handletextpad=0.2, handlelength=2., borderaxespad=0)
        plt.legend(handles=handles, **config['legend'])
        plt.gca().tick_params(**config['tick_params'])

        plt.subplots_adjust(**config['subplots_adjust'])
        plt.xticks(fontsize=config['fontsize'])
        plt.yticks(fontsize=config['fontsize'])

        plt.tight_layout()
        save_and_crop(fig, config['image_name'], dpi=900, bbox_inches='tight')
        plt.show()
        plt.close()
