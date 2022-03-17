#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import matplotlib.patches as mpatches

from plot.plotting_utilities import *

project_dir = './'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# csv_file = res_dir + 'csv/interp-kick1/all_results2.csv'

plots_config = {

    # 'plot5': {
    #     'files': {
    #         res_dir+'raw/PS-4MPPB-comb1-mtw50-r1-2/comm-comp-report.csv': {
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
    #         # 'comm': 'tab:green',
    #         # 'comp': 'tab:blue',
    #         # 'serial': 'tab:orange',
    #         # 'other': 'tab:purple'
    #         'comm': '0.2',
    #         'comp': '0.6',
    #         'serial': '1',
    #         # 'other': '1'

    #         # '10': 'tab:red'
    #         # '20000000-20-comp': 'tab:purple'
    #     },
    #     'hatches': {
    #         # '5': '/',
    #         '10': '',
    #         '20': 'x',
    #     },
    #     'order': ['comm', 'serial', 'comp'],
    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'x_name': 'n',
    #     'width': 3,
    #     'displs': 3.5,
    #     'omp_name': 'omp',
    #     'y_name': 'avg_percent',
    #     'y_err_name': 'std',
    #     'xlabel': 'Cores (x10)',
    #     'ylabel': 'Percent',
    #     'title': 'Run-Time breakdown',
    #     'ylim': [0, 100],
    #     'figsize': (5, 3),
    #     'image_name': images_dir + 'PS-4MPPB-comb1-mtw50-r1-2-newhisto.pdf'

    # },


    'plot4': {
        'files': {
            res_dir+'raw/LHC-96B-2MPPB-uint16/comm-comp-report.csv': {
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
            # 'comm': 'tab:green',
            # 'comp': 'tab:blue',
            # 'serial': 'tab:orange',
            # 'other': 'tab:purple'
            'comm': '0.2',
            'comp': '0.6',
            'serial': '1',
            # 'other': '1'

            # '10': 'tab:red'
            # '20000000-20-comp': 'tab:purple'
        },
        'hatches': {
            # '5': '/',
            '10': '',
            '20': 'x',
        },
        'order': ['comm', 'serial', 'comp'],
        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'x_name': 'n',
        'width': 3,
        'displs': 3.5,
        'omp_name': 'omp',
        'y_name': 'avg_percent',
        'y_err_name': 'std',
        'xlabel': 'Cores (x10)',
        'ylabel': 'Percent',
        'title': 'Run-Time breakdown',
        'ylim': [0, 100],
        'figsize': (5, 3),
        'image_name': images_dir + 'LHC-96B-2MPPB-uint16-newhisto.pdf'

    },



    # 'plot3': {
    #     'files': {
    #         res_dir+'raw/SPS-72B-4MPPB-uint16-r1-2/comm-comp-report.csv': {
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
    #         # 'comm': 'tab:green',
    #         # 'comp': 'tab:blue',
    #         # 'serial': 'tab:orange',
    #         # 'other': 'tab:purple'
    #         'comm': '0.2',
    #         'comp': '0.6',
    #         'serial': '1',
    #         # 'other': '1'

    #         # '10': 'tab:red'
    #         # '20000000-20-comp': 'tab:purple'
    #     },
    #     'hatches': {
    #         # '5': '/',
    #         '10': '',
    #         '20': 'x',
    #     },
    #     'order': ['comm', 'serial', 'comp'],
    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'x_name': 'n',
    #     'width': 3,
    #     'displs': 3.5,
    #     'omp_name': 'omp',
    #     'y_name': 'avg_percent',
    #     'y_err_name': 'std',
    #     'xlabel': 'Cores (x10)',
    #     'ylabel': 'Percent',
    #     'title': 'Run-Time breakdown',
    #     'ylim': [0, 100],
    #     'figsize': (5, 3),
    #     'image_name': images_dir + 'SPS-72B-4MPPB-uint16-r1-2-newhisto.pdf'

    # },



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
        # print(plots_dir)

        final_dir = {}

        for key in plots_dir.keys():
            omp = key.split('-')[0]
            phase = key.split('-')[1]
            if omp not in final_dir:
                final_dir[omp] = {}
            final_dir[omp][phase] = plots_dir[key]

        # print(final_dir)

        # exit()

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

        displs = 0.
        xticks = []
        for omp in final_dir.keys():
            bottom = None
            for phase in config['order']:
                values = final_dir[omp][phase]
            # for phase, values in final_dir[omp].items():
                if (phase in ['overhead', 'other']):
                    continue
                x = np.array(values[:, header.index(config['x_name'])], float)
                # omp = np.array(values[:, header.index(config['omp_name'])], float)
                x = (x-1) * int(omp)
                y = np.array(values[:, header.index(config['y_name'])], float)

                y_err = np.array(
                    values[:, header.index(config['y_err_name'])], float)

                if bottom is None:
                    bottom = np.empty_like(y)

                if len(x) > len(xticks):
                    xticks = x

                # if phase == 'serial':
                #     y += np.array(final_dir[omp]['overhead']
                #                   [:, header.index(config['y_name'])], float)
                #     y_err += np.array(final_dir[omp]['overhead']
                #                       [:, header.index(config['y_err_name'])], float)

                #     y += np.array(final_dir[omp]['other']
                #                   [:, header.index(config['y_name'])], float)
                #     y_err += np.array(final_dir[omp]['other']
                #                       [:, header.index(config['y_err_name'])], float)

                # print(bot, y)
                plt.bar(x + displs, y, width=config['width'],
                        bottom=bottom,
                        # yerr=y_err,
                        capsize=1, linewidth=.5,
                        edgecolor='black',
                        color=config['colors'][phase],
                        hatch=config['hatches'][omp])
                # label=phase)
                # color=config['colors'][label[0]])

                bottom += y

                if phase == 'serial' and omp=='10':
                    annotate(plt.gca(), x+displs, bottom, ha='center',
                        size=10.5)


            displs += config['displs']

        if 'extra' in config:
            for c in config['extra']:
                exec(c)

        handles = []
        for k, v in config['colors'].items():
            patch = mpatches.Patch(label=k, edgecolor='black', facecolor=v,
                                   linewidth=.5,)
            handles.append(patch)

        for k, v in config['hatches'].items():
            patch = mpatches.Patch(label=k + 'C/T', edgecolor='black',
                                   facecolor='.6', hatch=v, linewidth=.5,)
            handles.append(patch)

        # for k, v in config['markers'].items():
        #     line = mlines.Line2D(
        #         [], [], color='black', marker=v, label=k)
        # handles.append(line)

        plt.xticks(xticks+config['width']/2, np.array(xticks, int)//10)
        # plt.legend(loc='best', fancybox=True, fontsize=9.5,
        plt.legend(handles=handles, loc='center left', fancybox=True, fontsize=9,
                   ncol=1, columnspacing=1,
                   labelspacing=0.1, borderpad=0.5, framealpha=0.7,
                   handletextpad=0.2, handlelength=2., borderaxespad=0)
        # bbox_to_anchor=(0.1, 1.15))
        plt.tight_layout()
        save_and_crop(fig, config['image_name'], dpi=900, bbox_inches='tight')
        # plt.savefig(config['image_name'], dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

    # plt.legend(loc='best', fancybox=True, fontsize='11')
    # plt.axvline(700.0, color='k', linestyle='--', linewidth=1.5)
    # plt.axvline(1350.0, color='k', linestyle='--', linewidth=1.5)
    # plt.annotate('Heavy\nCombine\nWorkload', xy=(
    #     1400, 8.2), textcoords='data', size='16')
