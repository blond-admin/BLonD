#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os

from plot.plotting_utilities import *

project_dir = './'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# csv_file = res_dir + 'csv/interp-kick1/all_results2.csv'

plots_config = {

    'plot5': {
        'files': {
            res_dir+'raw/PS-2MPPB-r1/comm-comp-report.csv': {
                    'lines': {
                        'omp': ['2', '5', '10', '20'],
                        'type': ['comp', 'serial', 'comm', 'other', 'overhead']}
            }

        },
        'labels': {
            # '1': 'hyb-T1',
            # '2': 'hyb-T2',
            # '4': 'hyb-T4',
            '5': 'hyb-T5',
            '10': 'hyb-T10',
            '20': 'hybrid-T20'
        },
        'colors': {
            '1': 'tab:purple',
            '20': 'tab:blue',
            '4': 'tab:orange',
            '5': 'tab:green',
            '10': 'tab:red'
            # '20000000-20-comp': 'tab:purple'
        },
        'markers': {
            'comm': 'o',
            'comp': 'x',
            'serial': '^',
            'other': 's'
        },
        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'x_name': 'n',
        'omp_name': 'omp',
        'y_name': 'avg_percent',
        'y_err_name': 'std',
        'xlabel': 'MPI Tasks/Threads',
        'ylabel': 'Run-time percent',
        'title': 'MPI Time breakdown',
        'ylim': [0, 100],
        'figsize': (6, 3),
        'image_name': images_dir + 'PS-2MPPB-r1-histo.pdf'

    },

    # 'plot5': {
    #     'files': {
    #         res_dir+'raw/LHC-96B-uint16-r1/comm-comp-report.csv': {
    #                 'lines': {
    #                     'omp': ['2', '5', '10', '20'],
    #                     'type': ['comp', 'serial', 'comm', 'other', 'overhead']}
    #         }

    #     },
    #     'labels': {
    #         # '1': 'hyb-T1',
    #         # '2': 'hyb-T2',
    #         # '4': 'hyb-T4',
    #         '5': 'hyb-T5',
    #         '10': 'hyb-T10',
    #         '20': 'hybrid-T20'
    #     },
    #     'colors': {
    #         '1': 'tab:purple',
    #         '20': 'tab:blue',
    #         '4': 'tab:orange',
    #         '5': 'tab:green',
    #         '10': 'tab:red'
    #         # '20000000-20-comp': 'tab:purple'
    #     },
    #     'markers': {
    #         'comm': 'o',
    #         'comp': 'x',
    #         'serial': '^',
    #         'other': 's'
    #     },
    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_percent',
    #     'y_err_name': 'std',
    #     'xlabel': 'MPI Tasks/Threads',
    #     'ylabel': 'Run-time percent',
    #     'title': 'MPI Time breakdown',
    #     'ylim': [0, 100],
    #     'figsize': (6, 3),
    #     'image_name': images_dir + 'LHC-96B-uint16-r1-histo.pdf'

    # },

    # 'plot4': {
    #     'files': {
    #         res_dir+'raw/SPS-8n-72B-packed-mul-r2/comm-comp-report.csv': {
    #                 'lines': {
    #                     'omp': ['2', '5', '10', '20'],
    #                     'type': ['comp', 'serial', 'comm', 'other', 'overhead']}
    #         }

    #     },
    #     'labels': {
    #         # '1': 'hyb-T1',
    #         # '2': 'hyb-T2',
    #         # '4': 'hyb-T4',
    #         '5': 'hyb-T5',
    #         '10': 'hyb-T10',
    #         '20': 'hybrid-T20'
    #     },
    #     'colors': {
    #         '1': 'tab:purple',
    #         '20': 'tab:blue',
    #         '4': 'tab:orange',
    #         '5': 'tab:green',
    #         '10': 'tab:red'
    #         # '20000000-20-comp': 'tab:purple'
    #     },
    #     'markers': {
    #         'comm': 'o',
    #         'comp': 'x',
    #         'serial': '^',
    #         'other': 's'
    #     },
    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_percent',
    #     'y_err_name': 'std',
    #     'xlabel': 'MPI Tasks/Threads',
    #     'ylabel': 'Run-time percent',
    #     'title': 'MPI Time breakdown',
    #     'ylim': [0, 100],
    #     'figsize': (6, 3),
    #     'image_name': images_dir + 'SPS-8n-72B-packed-mul-r2-histo.pdf'

    # },

    # 'plot3': {
    #     'files': {
    #         res_dir+'raw/SPS-8n-72B-packed-mul-r5/comm-comp-report.csv': {
    #                 'lines': {
    #                     'omp': ['2', '5', '10', '20'],
    #                     'type': ['comp', 'serial', 'comm', 'other', 'overhead']}
    #         }

    #     },
    #     'labels': {
    #         # '1': 'hyb-T1',
    #         # '2': 'hyb-T2',
    #         # '4': 'hyb-T4',
    #         '5': 'hyb-T5',
    #         '10': 'hyb-T10',
    #         '20': 'hybrid-T20'
    #     },
    #     'colors': {
    #         '1': 'tab:purple',
    #         '20': 'tab:blue',
    #         '4': 'tab:orange',
    #         '5': 'tab:green',
    #         '10': 'tab:red'
    #         # '20000000-20-comp': 'tab:purple'
    #     },
    #     'markers': {
    #         'comm': 'o',
    #         'comp': 'x',
    #         'serial': '^',
    #         'other': 's'
    #     },
    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_percent',
    #     'y_err_name': 'std',
    #     'xlabel': 'MPI Tasks/Threads',
    #     'ylabel': 'Run-time percent',
    #     'title': 'MPI Time breakdown',
    #     'ylim': [0, 100],
    #     'figsize': (6, 3),
    #     'image_name': images_dir + 'SPS-8n-72B-packed-mul-r5-histo.pdf'

    # }

    # 'plot5': {
    #     'files': {
    #         res_dir+'raw/LHC-4n-96B-lt-lb-nogat-int-op-knd-r5-10kt/comm-comp-report.csv': {
    #             'lines': {
    #                 'omp': ['2', '4', '5', '10'],
    #                 'type': ['comp', 'serial', 'comm', 'other', 'overhead']}
    #         }

    #     },
    #     'labels': {
    #         # '1': 'hyb-T1',
    #         # '2': 'hyb-T2',
    #         '4': 'hyb-T4',
    #         '5': 'hyb-T5',
    #         '10': 'hyb-T10',
    #         # '20': 'hybrid-T20'
    #     },
    #     'colors': {
    #         '1': 'tab:purple',
    #         '2': 'tab:blue',
    #         '4': 'tab:orange',
    #         '5': 'tab:green',
    #         '10': 'tab:red'
    #         # '20000000-20-comp': 'tab:purple'
    #     },
    #     'markers': {
    #         'comm': 'o',
    #         'comp': 'x',
    #         'serial': '^',
    #         'other': 's'
    #     },
    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_percent',
    #     'y_err_name': 'std',
    #     'xlabel': 'MPI Tasks/Threads',
    #     'ylabel': 'Run-time percent',
    #     'title': 'MPI Time breakdown',
    #     'ylim': [0, 100],
    #     'figsize': (6, 3),
    #     'image_name': images_dir + 'LHC-histo-96B-lt-lb-nogat-int-op-knd-r5-10kt.pdf'

    # }


}

if __name__ == '__main__':
    for plot_key, config in plots_config.items():
        plots_dir = {}
        for file in config['files'].keys():
            print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header = list(data[0])
            data = data[1:]
            plots_dir.update(get_plots(header, data, config['files'][file]['lines'],
                                       exclude=config['files'][file].get('exclude', [])))
        print(plots_dir)

        fig = plt.figure(figsize=config['figsize'])
        plt.grid(True, which='major', alpha=0.6)
        plt.grid(True, which='minor', alpha=0.6, linestyle=':')
        # plt.minorticks_on()
        plt.title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        # plt.yscale('log', basex=2)
        if 'ylim' in config:
            plt.ylim(config['ylim'])

        for label, values in plots_dir.items():
            # print(values)
            if ('overhead' in label):
                continue
            x = np.array(values[:, header.index(config['x_name'])], float)
            omp = np.array(values[:, header.index(config['omp_name'])], float)
            x = (x-1) * omp
            y = np.array(values[:, header.index(config['y_name'])], float)
            # parts = np.array(values[:, header.index('parts')], float)
            # turns = np.array(values[:, header.index('turns')], float)
            # y = parts * turns / y
            y_err = np.array(
                values[:, header.index(config['y_err_name'])], float)
            # y_err = y_err * y / 100.
            print(label, x, y)
            # label = config['labels'][label]
            label = label.split('-')

            if (label[1] == 'others'):
                y += np.array(plots_dir[label[0] + '-overhead']
                              [:, header.index(config['y_name'])], float)
                y_err = np.array(
                    plots_dir[label[0] + '-overhead'][:, header.index(config['y_err_name'])], float)

            # if config['labels'][label[0]] in plt.gca().get_legend_handles_labels()[1]:
            plt.errorbar(x, y, yerr=y_err,
                         capsize=1, marker=config['markers'][label[1]], linewidth=1.5, elinewidth=1,
                         color=config['colors'][label[0]])
            # else:
            #     # print(config['colors'][])
            #     plt.errorbar(x, y, yerr=y_err, label=config['labels'][label[0]],
            #                  capsize=1, marker=config['markers'][label[1]], linewidth=1.5,  elinewidth=1,
            #                  color=config['colors'][label[0]])
            handles = []
            for k, v in config['labels'].items():
                line = mlines.Line2D([], [], color=config['colors'][k], marker='',
                                     label=v)
                handles.append(line)

            for k, v in config['markers'].items():
                line = mlines.Line2D([], [], color='black', marker=v, label=k)
                handles.append(line)

            # plt.legend(handles=handles)

        if 'extra' in config:
            for c in config['extra']:
                exec(c)

        # plt.legend(loc='best', fancybox=True, fontsize=9.5,
        plt.legend(handles=handles, loc='best', fancybox=True, fontsize=9,
                   ncol=2, columnspacing=1,
                   labelspacing=0.1, borderpad=0.2, framealpha=0.5,
                   handletextpad=0.2, handlelength=1.5, borderaxespad=0)
        # bbox_to_anchor=(0.1, 1.15))
        plt.tight_layout()
        save_and_crop(fig, config['image_name'], dpi=600, bbox_inches='tight')
        # plt.savefig(config['image_name'], dpi=600, bbox_inches='tight')
        # subprocess.call
        plt.show()
        plt.close()

    # plt.legend(loc='best', fancybox=True, fontsize='11')
    # plt.axvline(700.0, color='k', linestyle='--', linewidth=1.5)
    # plt.axvline(1350.0, color='k', linestyle='--', linewidth=1.5)
    # plt.annotate('Heavy\nCombine\nWorkload', xy=(
    #     1400, 8.2), textcoords='data', size='16')
