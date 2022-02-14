import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines
import matplotlib.ticker

from plot.plotting_utilities import *

project_dir = './'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)


plots_config = {

    'plot1': {
        'files': {

            res_dir+'raw/SPS/b72-4MPPB-t10k-2/comm-comp-report.csv': {
                'key': 'exact',
                # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
                'lines': {
                    'omp': ['10', '20'],
                    'type': ['total']
                }
            },
            # res_dir+'raw/SPS-b72-4MPPB-t10k-approx-time-2/comm-comp-report.csv': {
            #     'key': 'method2',
            #     # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
            #     'lines': {
            #         'omp': ['10', '20'],
            #         'type': ['total']
            #     }
            # },

            res_dir+'raw/SPS/b72-4MPPB-t10k-red2-time-2/comm-comp-report.csv': {
                'key': '2turns',
                # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
                'lines': {
                    'omp': ['10', '20'],
                    'type': ['total']
                }
            },
            res_dir+'raw/SPS/b72-4MPPB-t10k-red3-time-3/comm-comp-report.csv': {
                'key': '3turns',
                # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
                'lines': {
                    'omp': ['10', '20'],
                    'type': ['total']
                }
            },
            res_dir+'raw/SPS/b72-4MPPB-t10k-red4-time-3/comm-comp-report.csv': {
                'key': '4turns',
                # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
                'lines': {
                    'omp': ['10', '20'],
                    'type': ['total']
                }
            },
        },
        'labels': {
            'exact': 'exact',
            '2turns': '2turns',
            '3turns': '3turns',
            '4turns': '4turns',
            'method2': 'method2',
            'SPS-t43k': 'SPS-t43k',
            'SPS-t4k': 'SPS-t4k',
            'SPS-no-approx': 'SPS-no-approx',
            'PS-t100k': 'PS-t100k',
            'PS-t10k': 'PS-t10k',
            'PS-no-approx': 'PS-no-approx',
            'LHC-t100k': 'LHC-t100k',
            'LHC-t10k': 'LHC-t10k',
            'LHC-no-approx': 'LHC-no-approx',
            '10-total': '10 Cores/Task',
            '20-total': '20 Cores/Task'
        },
        'markers': {
            'exact': 's',
            '2turns': 'o',
            '3turns': 'o',
            '4turns': 'o',
            'method2': 'o'
        },
        'colors': {
            'exact': 'black',
            '2turns': 'tab:blue',
            '3turns': 'tab:orange',
            '4turns': 'tab:green',
            'method2': 'tab:blue',
            '10-total': 'tab:blue',
            '20-total': 'tab:orange',
            'SPS-t43k': 'tab:blue',
            'SPS-t4k': 'tab:blue',
            'SPS-no-approx': 'tab:blue',
            'PS-t100k': 'tab:orange',
            'PS-t10k': 'tab:orange',
            'PS-no-approx': 'tab:orange',
            'LHC-t100k': 'tab:brown',
            'LHC-t10k': 'tab:brown',
            'LHC-no-approx': 'tab:brown',
            # '': 'tab:red'
        },
        'ls': {
            'exact': '-',
            '2turns': '--',
            '3turns': '--',
            '4turns': '--',
            'method2': '--',
            '10-total': '-',
            '20-total': '-',
            'SPS-t43k': '-',
            'SPS-t4k': '--',
            'SPS-no-approx': ':',
            'PS-t100k': '-',
            'PS-t10k': '--',
            'PS-no-approx': ':',
            'LHC-t100k': '-',
            'LHC-t10k': '-',
            'LHC-no-approx': ':',
            # '': 'tab:red'
        },
        'reference': {
            'exact': {'time': 430., 'parts': 4000000, 'turns': 100},
            '2turns': {'time': 430., 'parts': 4000000, 'turns': 100},
            '3turns': {'time': 430., 'parts': 4000000, 'turns': 100},
            '4turns': {'time': 430., 'parts': 4000000, 'turns': 100},
            'method2': {'time': 430., 'parts': 4000000, 'turns': 100},
            '10-total': {'time': 430., 'parts': 4000000, 'turns': 100},
            '20-total': {'time': 430., 'parts': 4000000, 'turns': 100},
            'SPS-t43k': {'time': 430., 'parts': 4000000, 'turns': 100},
            'SPS-t4k': {'time': 430., 'parts': 4000000, 'turns': 100},
            'SPS-no-approx': {'time': 430., 'parts': 4000000, 'turns': 100},
            'LHC-t100k': {'time': 2120., 'parts': 2000000, 'turns': 1000},
            'LHC-t10k': {'time': 2120., 'parts': 2000000, 'turns': 1000},
            'LHC-no-approx': {'time': 2120., 'parts': 2000000, 'turns': 1000},
            'PS-t100k': {'time': 1623.7, 'parts': 4000000, 'turns': 2000},
            'PS-t10k': {'time': 1623.7, 'parts': 4000000, 'turns': 2000},
            'PS-no-approx': {'time': 1623.7, 'parts': 4000000, 'turns': 2000}
        },
        # 'reference': {'time': 430., 'parts': 4000000, 'turns': 100},

        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'x_name': 'n',
        'omp_name': 'omp',
        'y_name': 'avg_time(sec)',
        # 'y_err_name': 'std',
        'xlabel': 'Cores (x10)',
        'ylabel': 'Speedup',
        'title': '',
        'ylim': {
            'speedup': [0, 140]
        },
        'nticks': 6,
        'legend_loc': 'upper left',
        'figsize': (4, 4),
        'image_name': images_dir + 'SPS-speedup-liu-method1-with-two-axis.pdf'

    },

    # 'plot4': {
    #     'files': {
    #         res_dir+'raw/LHC-96B-2MPPB-uint16-nobcast-r1-2/comm-comp-report.csv': {
    #             'lines': {
    #                 'omp': ['2', '5', '10', '20'],
    #                 'type': ['total']}
    #         }

    #     },
    #     'labels': {
    #         '1-total': '1C/T',
    #         '2-total': '2C/T',
    #         '4-total': '4C/T',
    #         '5-total': '5C/T',
    #         '10-total': '10C/T',
    #         '20-total': '20C/T'
    #     },
    #     'markers': {
    #         # '5-total': 'x',
    #         '10-total': 's',
    #         '20-total': 'o'
    #     },
    #     'colors': {
    #         'speedup': 'tab:blue',
    #         'efficiency': 'tab:red'
    #     },
    #     # 'reference': {'time': 200.71, 'parts': 2000000, 'turns': 100},
    #     'reference': {'time': 2120., 'parts': 2000000, 'turns': 1000},

    #     # 'reference': { 'time': 8213. , 'parts': 1000000, 'turns':10000},

    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'ideal': '2-total',
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_time(sec)',
    #     # 'y_err_name': 'std',
    #     'xlabel': 'Cores (x10)',
    #     'ylabel': ['Speedup', 'Efficiency'],
    #     'title': 'Speedup-Efficiency graph',
    #     'ylim': {
    #         'speedup': [0, 120],
    #         'efficiency': [60, 120]
    #     },
    #     'nticks': 7,
    #     'legend_loc':'lower center',
    #     'figsize': (5, 3),
    #     'image_name': images_dir + 'LHC-96B-2MPPB-uint16-nobcast-r1-2-speedup.pdf'

    # },



    # 'plot2': {
    #     'files': {
    #         res_dir+'raw/SPS-72B-4MPPB-uint16-r1-2/comm-comp-report.csv': {
    #             'lines': {
    #                 'omp': ['2', '5', '10', '20'],
    #                 'type': ['total']}
    #         }

    #     },

    #     'reference': {'time': 430., 'parts': 4000000, 'turns': 100},
    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'labels': {
    #         '1-total': '1C/T',
    #         '2-total': '2C/T',
    #         '4-total': '4C/T',
    #         '5-total': '5C/T',
    #         '10-total': '10C/T',
    #         '20-total': '20C/T'
    #     },
    #     'markers': {
    #         # '5-total': 'x',
    #         '10-total': 's',
    #         '20-total': 'o'
    #     },
    #     'colors': {
    #         'speedup': 'tab:blue',
    #         'efficiency': 'tab:red'
    #     },

    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'ideal': '2-total',
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_time(sec)',
    #     # 'y_err_name': 'std',
    #     'xlabel': 'Cores (x10)',
    #     'ylabel': ['Speedup', 'Efficiency'],
    #     'title': 'Speedup-Efficiency graph',
    #     'ylim': {
    #         'speedup': [0, 120],
    #         'efficiency': [60, 150]
    #     },
    #     'nticks': 7,
    #     'legend_loc':'lower center',
    #     'figsize': (5, 3),
    #     'image_name': images_dir + 'SPS-72B-4MPPB-uint16-r1-2-speed-eff.pdf'

    # },


}

if __name__ == '__main__':
    for plot_key, config in plots_config.items():
        plots_dir = {}
        for file in config['files'].keys():
            # print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header = list(data[0])
            data = data[1:]
            temp = get_plots(header, data, config['files'][file]['lines'],
                             exclude=config['files'][file].get('exclude', []))
            temp[config['files'][file]['key']] = temp['10-total']
            del temp['10-total']
            plots_dir.update(temp)

        # print(plots_dir)
        fig = plt.figure(figsize=config['figsize'])
        # ax1 = fig.add_subplot(111)
        # ax2 = ax1.twinx()
        ax1 = plt.gca()

        plt.grid(True, which='major', axis='y', alpha=1)
        plt.grid(False, which='both', axis='x', alpha=0)

        # plt.grid(True, which='minor', alpha=0.6, linestyle=':')
        # plt.minorticks_on()
        plt.title(config['title'])
        # ax1.set_title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'], color='tab:blue')
        # plt.ylim(config['ylim']['speedup'])
        # , size='12', weight='semibold')

        # plt.yscale('log', basex=2)
        # if 'ylim' in config:
        #     plt.ylim(config['ylim'])

        for key, values in plots_dir.items():
            # print(values)
            label = config['labels'][key]
            x = np.array(values[:, header.index(config['x_name'])], float)
            omp = np.array(
                values[:, header.index(config['omp_name'])], float)
            x = (x) * omp

            y = np.array(values[:, header.index(config['y_name'])], float)
            parts = np.array(values[:, header.index('parts')], float)
            turns = np.array(values[:, header.index('turns')], float)
            # This is the throughput
            y = parts * turns / y

            # Now the reference, 1thread
            yref = config['reference'][key]['time']
            partsref = config['reference'][key]['parts']
            turnsref = config['reference'][key]['turns']
            yref = partsref * turnsref / yref

            speedup = y / yref

            # efficiency = 100 * speedup / x

            # We want speedup, compared to 1 worker with 1 thread
            if key == 'exact':
                color = 'black'
            else:
                color = 'tab:blue'
            plt.errorbar(x//10, speedup, yerr=None, 
                         # color=config['colors'][key],
                         color=color,
                         capsize=2, marker=config['markers'][key], markersize=4,
                         linewidth=2., label=label,
                         ls=config['ls'][key])

            # if '10' in key:
            #     plt.xticks(x//10)
            annotate_max(plt.gca(), x//10, speedup, ha='center', va='bottom',
                         size='10')

            # ax2.errorbar(x//10, efficiency, yerr=None, color=config['colors']['efficiency'],
            #              capsize=2, marker=config['markers'][key], markersize=4,
            #              linewidth=1.)

        if 'extra' in config:
            for c in config['extra']:
                exec(c)

        # nticks = config['nticks']
        # plt.gca().yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
        # ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
        # for tl in ax1.get_yticklabels():
        #     tl.set_color(config['colors']['speedup'])

        # handles = []
        # for k, v in config['markers'].items():
        #     line = mlines.Line2D([], [], color='black',
        #                          marker=v, label=config['labels'][k])
        #     handles.append(line)

        ax2 = ax1.twinx()
        plt.sca(ax2)
        plt.ylabel('Speedup over exact', color='tab:red')
        
        # Get the exact throughput

        valexact = plots_dir['exact']
        yexact = np.array(valexact[:, header.index(config['y_name'])], float)
        parts = np.array(valexact[:, header.index('parts')], float)
        turns = np.array(valexact[:, header.index('turns')], float)
        # This is the throughput
        yexact = parts * turns / yexact

        for key, values in plots_dir.items():
            # print(values)
            if key == 'exact':
                continue

            label = config['labels'][key]
            x = np.array(values[:, header.index(config['x_name'])], float)
            omp = np.array(
                values[:, header.index(config['omp_name'])], float)
            x = (x) * omp

            y = np.array(values[:, header.index(config['y_name'])], float)
            parts = np.array(values[:, header.index('parts')], float)
            turns = np.array(values[:, header.index('turns')], float)
            # This is the throughput
            y = parts * turns / y

            # speedup = 100. * (y - yexact) / yexact
            speedup = (y) / yexact

            # efficiency = 100 * speedup / x

            # We want speedup, compared to 1 worker with 1 thread
            plt.errorbar(x//10, speedup, yerr=None, 
                         color='tab:red',
                         capsize=2, marker=config['markers'][key], markersize=4,
                         linewidth=2., 
                         # label=label,
                         ls=config['ls'][key])

            # annotate_max(plt.gca(), x//10, speedup, ha='center', va='bottom',
            #              size='10')

        for tl in ax2.get_yticklabels():
            tl.set_color('tab:red')
        for tl in ax1.get_yticklabels():
            tl.set_color('tab:blue')
        plt.sca(ax1)
        plt.xticks(x//10)
        plt.legend(loc=config['legend_loc'], fancybox=True, fontsize=10.5,
                   labelspacing=0, borderpad=0.5, framealpha=0.4,
                   handletextpad=0.5, handlelength=2, borderaxespad=0)
        plt.tight_layout()
        save_and_crop(fig, config['image_name'], dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

    # plt.legend(loc='best', fancybox=True, fontsize='11')
    # plt.axvline(700.0, color='k', linestyle='--', linewidth=1.5)
    # plt.axvline(1350.0, color='k', linestyle='--', linewidth=1.5)
    # plt.annotate('Light\nCombine\nWorkload', xy=(
    #     200, 6.3), textcoords='data', size='16')
    # plt.annotate('Moderate\nCombine\nWorkload', xy=(
    #     800, 6.3), textcoords='data', size='16')
    # plt.annotate('Heavy\nCombine\nWorkload', xy=(
    #     1400, 8.2), textcoords='data', size='16')
