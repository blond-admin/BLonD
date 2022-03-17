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

    'plot3': {
         'files': {
            res_dir+'raw/SPS-72B-4MPPB-uint16-r1-2/comm-comp-report.csv': {
                'key': 'r1',
                'lines': {
                    'omp': ['10'],
                    'type': ['serial', 'comm', 'other', 'overhead']
                }
            },
            res_dir+'raw/SPS-72B-4MPPB-uint16-r2-2/comm-comp-report.csv': {
                'key': 'r2',
                'lines': {
                    'omp': ['10'],
                    'type': ['serial', 'comm', 'other', 'overhead']
                }
            },
            # res_dir+'raw/SPS-72B-4MPPB-uint16-r3-2/comm-comp-report.csv': {
            #     'key': 'r3',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['serial', 'comm', 'other', 'overhead']
            #     }
            # },
            res_dir+'raw/SPS-72B-4MPPB-uint16-r4-2/comm-comp-report.csv': {
                'key': 'r4',
                'lines': {
                    'omp': ['10'],
                    'type': ['serial', 'comm', 'other', 'overhead']
                }
            }

        },
        'labels': {
            'r1': 'r1',
            'r2': 'r2',
            'r3': 'r3',
            'r4': 'r4'
        },
        'colors': {
            'comm': '0.6',
            # 'comp': '0.2',
            'serial': '1',
        },
        'hatches': {
            # '5': '/',
            'r1': '',
            'r2': '.',
            # 'r3': '/',
            'r4': 'x'
        },
        'omp': 10,
        'order': ['comm', 'serial'],
        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'x_name': 'n',
        'width': 2.6,
        'displs': 2.65,
        'omp_name': 'omp',
        'y_name': 'avg_percent',
        'y_err_name': 'std',
        'xlabel': 'Cores (x10)',
        'ylabel': 'Percent',
        'title': 'Run-Time breakdown',
        'ylim': [0, 55],
        'figsize': (5, 3),
        'image_name': images_dir + 'SPS-72B-4MPPB-uint16-multi-reduce-histo.pdf'

    },



}

if __name__ == '__main__':
    for plot_key, config in plots_config.items():
        plots_dir = {}
        for file in config['files'].keys():
            print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header = list(data[0])
            data = data[1:]
            temp = get_plots(header, data, config['files'][file]['lines'],
                             exclude=config['files'][file].get('exclude', []))
            temp2 = {}
            for k in temp.keys():
                temp2[k+'-'+config['files'][file]['key']] = temp[k]                
            plots_dir.update(temp2)
        # print(plots_dir)

        final_dir = {}

        for key in plots_dir.keys():
            omp, phase, red = key.split('-')

            if red not in final_dir:
                final_dir[red] = {}
            final_dir[red][phase] = plots_dir[key]

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
        for red in final_dir.keys():
            bottom = None
            for phase in config['order']:
                values = final_dir[red][phase]
                if (phase in ['overhead', 'other']):
                    continue
                x = np.array(values[:, header.index(config['x_name'])], float)
                x = (x-1) * int(config['omp'])
                y = np.array(values[:, header.index(config['y_name'])], float)

                y_err = np.array(
                    values[:, header.index(config['y_err_name'])], float)

                if bottom is None:
                    bottom = np.zeros(len(y))

                if len(x) > len(xticks):
                    xticks = x

                if phase == 'serial':
                    y += np.array(final_dir[red]['overhead']
                                  [:, header.index(config['y_name'])], float)
                    y_err += np.array(final_dir[red]['overhead']
                                      [:, header.index(config['y_err_name'])], float)

                    y += np.array(final_dir[red]['other']
                                  [:, header.index(config['y_name'])], float)
                    y_err += np.array(final_dir[red]['other']
                                      [:, header.index(config['y_err_name'])], float)

                # print(bot, y)
                plt.bar(x + displs, y, width=config['width'],
                        bottom=bottom,
                        # yerr=y_err,
                        capsize=1, linewidth=.5,
                        edgecolor='black',
                        color=config['colors'][phase],
                        hatch=config['hatches'][red])
                # label=phase)
                # color=config['colors'][label[0]])

                bottom += y

                # if phase == 'serial' and omp=='10':
                #     annotate(plt.gca(), x+displs, bottom, ha='center',
                #         size=10.5)


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
            patch = mpatches.Patch(label=k, edgecolor='black',
                                   facecolor='.6', hatch=v, linewidth=.5,)
            handles.append(patch)

        # for k, v in config['markers'].items():
        #     line = mlines.Line2D(
        #         [], [], color='black', marker=v, label=k)
        # handles.append(line)

        plt.xticks(xticks+config['width'], np.array(xticks, int)//10)
        # plt.legend(loc='best', fancybox=True, fontsize=9.5,
        plt.legend(handles=handles, loc='upper left', fancybox=True, fontsize=10,
                   ncol=2, columnspacing=1,
                   labelspacing=0.1, borderpad=0.5, framealpha=0.7,
                   handletextpad=0.2, handlelength=3., borderaxespad=0)
        # bbox_to_anchor=(0.1, 1.15))
        plt.xlim((5 ,160))
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
