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
            # res_dir+'raw/SPS-b72-4MPPB-approx-time/comm-comp-report.csv': {
            #     'key': 'SPS-t4k',
            #     # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },

            res_dir+'raw/SPS-b72-4MPPB-t43k-approx-time/comm-comp-report.csv': {
                'key': 'SPS-t43k',
                # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },


            res_dir+'raw/SPS-b72-4MPPB-t10k/comm-comp-report.csv': {
                'key': 'SPS-no-approx',
                # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },

            # res_dir+'raw/SPS-72B-4MPPB-uint16-r2-2/comm-comp-report.csv': {
            #     'key': 'red-2',
            #     # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },

            # res_dir+'raw/SPS-72B-4MPPB-uint16-r3-2/comm-comp-report.csv': {
            #     'key': 'red-3',
            #     # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },

            # res_dir+'raw/SPS-72B-4MPPB-uint16-r4-2/comm-comp-report.csv': {
            #     'key': 'red-4',
            #     # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },



            # res_dir+'raw/SPS-b1-4MPPB-approx-time/comm-comp-report.csv': {
            #     'key': 'SPS',
            #     # 'reference':  {'time': 430., 'parts': 4000000, 'turns': 100},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            # res_dir+'raw/PS-b21-approx-t100k-time/comm-comp-report.csv': {
            #     'key': 'PS-t100k',
            #     # 'reference':  {'time': 1623.7, 'parts': 4000000, 'turns': 2000},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },

            # res_dir+'raw/PS-b21-approx-time/comm-comp-report.csv': {
            #     'key': 'PS-t10k',
            #     # 'reference':  {'time': 1623.7, 'parts': 4000000, 'turns': 2000},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            # res_dir+'raw/PS-4MPPB-comb-mtw50/comm-comp-report.csv': {
            #     'key': 'PS-no-approx',
            #     # 'reference':  {'time': 1623.7, 'parts': 4000000, 'turns': 2000},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },

            # res_dir+'raw/LHC-b96-2MPPB-t100k-approx-time/comm-comp-report.csv': {
            #     'key': 'LHC-t100k',
            #     # 'reference':  {'time': 2120., 'parts': 2000000, 'turns': 1000},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },

            # res_dir+'raw/LHC-b96-2MPPB-approx-time/comm-comp-report.csv': {
            #     'key': 'LHC-t10k',
            #     # 'reference':  {'time': 2120., 'parts': 2000000, 'turns': 1000},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            # res_dir+'raw/LHC-96B-2MPPB-t10k/comm-comp-report.csv': {
            #     'key': 'LHC-no-approx',
            #     # 'reference':  {'time': 2120., 'parts': 2000000, 'turns': 1000},
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },

        },
        'labels': {
            'SPS-t43k': 'method2',
            'SPS-t4k': 'SPS-t4k',
            # 'SPS-no-approx': 'SPS-no-approx',
            'SPS-no-approx': 'exact',
            'red-2': 'red-2',
            'red-3': 'red-3',
            'red-4': 'red-4',
            'PS-t100k': 'PS-t100k',
            'PS-t10k': 'PS-t10k',
            'PS-no-approx': 'PS-no-approx',
            'LHC-t100k': 'LHC-t100k',
            'LHC-t10k': 'LHC-t10k',
            'LHC-no-approx': 'LHC-no-approx',
        },
        # 'markers': {
        #     '10-total': 's',
        #     '20-total': 'o'
        # },
        'colors': {
            'SPS-t43k': 'tab:blue',
            'SPS-t4k': 'tab:blue',
            'SPS-no-approx': 'black',
            'red-2': 'tab:blue',
            'red-3': 'tab:blue',
            'red-4': 'tab:blue',
            'PS-t100k': 'tab:orange',
            'PS-t10k': 'tab:orange',
            'PS-no-approx': 'tab:orange',
            'LHC-t100k': 'tab:brown',
            'LHC-t10k': 'tab:brown',
            'LHC-no-approx': 'tab:brown',
            # '': 'tab:red'
        },
        'ls': {
            'SPS-t43k': '-',
            'SPS-t4k': '--',
            'SPS-no-approx': ':',
            'red-2': ':',
            'red-3': ':',
            'red-4': ':',
            'PS-t100k': '-',
            'PS-t10k': '--',
            'PS-no-approx': ':',
            'LHC-t100k': '-',
            'LHC-t10k': '-',
            'LHC-no-approx': ':',
            # '': 'tab:red'
        },
        'reference': {
            'SPS-t43k': {'time': 430., 'parts': 4000000, 'turns': 100},
            'SPS-t4k': {'time': 430., 'parts': 4000000, 'turns': 100},
            'SPS-no-approx': {'time': 430., 'parts': 4000000, 'turns': 100},
            'red-2': {'time': 430., 'parts': 4000000, 'turns': 100},
            'red-3': {'time': 430., 'parts': 4000000, 'turns': 100},
            'red-4': {'time': 430., 'parts': 4000000, 'turns': 100},
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
        # 'ylim': {
        #     'speedup': [0, 140]
        # },
        'nticks': 6,
        'legend_loc': 'upper left',
        'figsize': (5, 3),
        'image_name': images_dir + 'sps-all-approx.pdf'

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

        plt.grid(True, which='major', axis='y', alpha=1)
        plt.grid(False, which='both', axis='x', alpha=0)

        # plt.grid(True, which='minor', alpha=0.6, linestyle=':')
        # plt.minorticks_on()
        plt.title(config['title'])
        # ax1.set_title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        if 'ylim' in config:
            plt.ylim(config['ylim']['speedup'])
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
            plt.errorbar(x//10, speedup, yerr=None, color=config['colors'][key],
                         capsize=2, marker=None, markersize=4,
                         linewidth=2., label=label,
                         ls=config['ls'][key])

            # if '10' in key:
            #     plt.xticks(x//10)
            # annotate_max(plt.gca(), x//10, speedup, ha='center', va='bottom',
            # size='9')

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
