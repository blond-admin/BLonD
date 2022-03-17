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
images_dir = res_dir + 'plots/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# csv_file = res_dir + 'csv/interp-kick1/all_results2.csv'

plots_config = {

    'plot4': {
        'files': {
            res_dir+'raw/EX01-mpich3/particles-report.csv': {
                    'key': 'ex01-mpich3',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
            res_dir+'raw/EX01-lb-mpich3/particles-report.csv': {
                    'key': 'ex01-lbmpich3',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },

            res_dir+'raw/EX01-mpich3-approx2/particles-report.csv': {
                    'key': 'ex01-mpich3apprx',
                    'lines': {
                        # 'omp': ['10'],'
                        'function': None}
            },
            res_dir+'raw/EX01-lb-mpich3-approx2/particles-report.csv': {
                    'key': 'ex01-lbmpich3apprx',
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
            'comp': '0.95',
            'serial': '0.7',
            'comm': '0.45',
            'redistr': '0.25',
        },
        'hatches': {
            'ex01': '/',
            # 'lhc': '/',
            # 'sps': 'o',
            # 'ps': 'x',

            # 'mpich3': '/',
            # 'openmpi3': 'o',
            # 'mvapich2': 'x',
        },
        'edgecolor': {
            'mpich3': 'xkcd:light blue',
            'lbmpich3': 'xkcd:blue',
            'mpich3apprx': 'xkcd:light green',
            'lbmpich3apprx': 'xkcd:green',
            # 'lbmpich32': 'xkcd:light orange',
            # 'lbmpich3apprx2': 'xkcd:orange',

        },

        # 'order': ['comm', 'serial', 'comp'],
        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'x_name': 'n',
        # 'width': 3,
        # 'displs': 3.5,
        'omp_name': 'omp',
        'y_name': 'global_percentage',
        'y_err_name': 'std',
        'xlabel': 'Cores (x10)',
        'ylabel': 'Percent',
        'title': 'Run-Time breakdown',
        'ylim': [0, 100],
        'figsize': (5, 3),
        'image_name': images_dir + 'redistr-bar-1.pdf'

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
                final_dir[tc][mpi][func]['y'] += get_values(plots_dir[key], header, config['y_name'])
            else:
                final_dir[tc][mpi][func] = {}
                final_dir[tc][mpi][func]['y'] = get_values(plots_dir[key], header, config['y_name'])
                final_dir[tc][mpi][func]['x'] = get_values(plots_dir[key], header, config['x_name'])

        # print(final_dir)

        fig = plt.figure(figsize=config['figsize'])
        plt.grid(False, axis='y')
        # plt.grid(True, which='major', alpha=0.6)
        # plt.grid(True, which='minor', alpha=0.6, linestyle=':')
        # plt.minorticks_on()
        plt.title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        # plt.yscale('log', basex=2)
        if 'ylim' in config:
            plt.ylim(config['ylim'])

        pos = 0.
        width=0.15
        intra_step = width
        inter_step = width/2.5

        for tc in final_dir.keys():
            for mpiv in final_dir[tc].keys():
                # funcs = list(final_dir[tc][mpiv].keys())
                # funcs.sort()
                bottom = None
                for f in ['comp', 'serial', 'comm', 'redistr']:
                    if f not in final_dir[tc][mpiv]:
                        continue
                    x = final_dir[tc][mpiv][f]['x']
                    y = final_dir[tc][mpiv][f]['y']
                    if bottom is None:
                        bottom = np.zeros(len(y))
                    plt.bar(np.arange(len(x)) + pos, y,
                            width=width,
                            bottom=bottom,
                            capsize=1, linewidth=1.,
                            edgecolor=config['edgecolor'][mpiv],
                            color=config['colors'][f],
                            hatch=config['hatches'][tc])
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
        plt.legend(handles=handles, loc='lower left', fancybox=True,
                   fontsize=9, ncol=2, columnspacing=1,
                   labelspacing=0.1, borderpad=0.5, framealpha=0.7,
                   handletextpad=0.2, handlelength=2., borderaxespad=0)
        # bbox_to_anchor=(0.1, 1.15))
        plt.tight_layout()
        save_and_crop(fig, config['image_name'], dpi=900, bbox_inches='tight')
        plt.show()
        plt.close()
