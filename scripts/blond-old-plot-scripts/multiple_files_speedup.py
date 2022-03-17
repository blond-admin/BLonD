import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines
import matplotlib.ticker
import sys
from plot.plotting_utilities import *

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)


plots_config = {

    # 'plot1': {
    #     'files': {
    #         res_dir+'raw/SPS-72B-4MPPB-uint16-r1-2/comm-comp-report.csv': {
    #             'key': 'r1',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },
    #         res_dir+'raw/SPS-72B-4MPPB-uint16-r2-2/comm-comp-report.csv': {
    #             'key': 'r2',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },
    #         res_dir+'raw/SPS-72B-4MPPB-uint16-r3-2/comm-comp-report.csv': {
    #             'key': 'r3',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         },
    #         res_dir+'raw/SPS-72B-4MPPB-uint16-r4-2/comm-comp-report.csv': {
    #             'key': 'r4',
    #             'lines': {
    #                 'omp': ['10'],
    #                 'type': ['total']
    #             }
    #         }

    #     },
    #     'labels': {
    #         'r1': 'every-turn',
    #         'r2': 'every-2-turns',
    #         'r3': 'every-3-turns',
    #         'r4': 'every-4-turns'
    #     },
    #     # 'markers': {
    #     #     '10-total': 's',
    #     #     '20-total': 'o'
    #     # },
    #     'colors': {
    #         'r1': 'tab:blue',
    #         'r2': 'tab:orange',
    #         'r3': 'tab:green',
    #         'r4': 'tab:red'
    #     },
    #     'reference': {'time': 430., 'parts': 4000000, 'turns': 100},

    #     # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
    #     'x_name': 'n',
    #     'omp_name': 'omp',
    #     'y_name': 'avg_time(sec)',
    #     # 'y_err_name': 'std',
    #     'xlabel': 'Cores (x10)',
    #     'ylabel': 'Speedup',
    #     'title': 'SPS Testcase',
    #     'ylim': {
    #         'speedup': [0, 210]
    #     },
    #     'nticks': 6,
    #     'legend_loc': 'upper left',
    #     'figsize': (4, 4),
    #     'image_name': images_dir + 'SPS-72B-4MPPB-uint16-multi-reduce.pdf'

    # },

    'plot1': {
        'files': {
            res_dir+'raw/SPS-b72-4MPPB-t10k-mpich3/comm-comp-report.csv': {
                'key': 'sps-mpich3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            # res_dir+'raw/SPS-b72-4MPPB-t10k/comm-comp-report.csv': {
            #     'key': 'sps-orig',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            res_dir+'raw/SPS-b72-4MPPB-t10k-mvapich2/comm-comp-report.csv': {
                'key': 'sps-mvapich2',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/SPS-b72-4MPPB-t10k-openmpi3/comm-comp-report.csv': {
                'key': 'sps-openmpi3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/LHC-96B-2MPPB-t10k-mpich3/comm-comp-report.csv': {
                'key': 'lhc-mpich3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            # res_dir+'raw/LHC-96B-2MPPB-t10k/comm-comp-report.csv': {
            #     'key': 'lhc-orig',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
            res_dir+'raw/LHC-96B-2MPPB-t10k-openmpi3/comm-comp-report.csv': {
                'key': 'lhc-openmpi3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/LHC-96B-2MPPB-t10k-mvapich2/comm-comp-report.csv': {
                'key': 'lhc-mvapich2',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/PS-b21-t10k-mpich3/comm-comp-report.csv': {
                'key': 'ps-mpich3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/PS-b21-t10k-openmpi3/comm-comp-report.csv': {
                'key': 'ps-openmpi3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/PS-b21-t10k-mvapich2/comm-comp-report.csv': {
                'key': 'ps-mvapich2',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            # res_dir+'raw/PS-4MPPB-comb-mtw50/comm-comp-report.csv': {
            #     'key': 'ps-orig',
            #     'lines': {
            #         'omp': ['10'],
            #         'type': ['total']
            #     }
            # },
        },
        'labels': {
            'lhc-mpich3': 'lhc-mpich3',
            'lhc-orig': 'lhc-orig',
            'lhc-openmpi3': 'lhc-openmpi3',
            'lhc-mvapich2': 'lhc-mvapich2',
            'sps-mpich3': 'sps-mpich3',
            'sps-orig': 'sps-orig',
            'sps-openmpi3': 'sps-openmpi3',
            'sps-mvapich2': 'sps-mvapich2',
            'ps-mpich3': 'ps-mpich3',
            'ps-orig': 'ps-orig',
            'ps-openmpi3': 'ps-openmpi3',
            'ps-mvapich2': 'ps-mvapich2',
        },
        'markers': {
            'lhc-mpich3': 'o',
            'lhc-orig': 'o',
            'lhc-openmpi3': 'o',
            'lhc-mvapich2': 'o',
            'sps-mpich3': 's',
            'sps-orig': 's',
            'sps-openmpi3': 's',
            'sps-mvapich2': 's',
            'ps-orig': 'x',
            'ps-mpich3': 'x',
            'ps-openmpi3': 'x',
            'ps-mvapich2': 'x',
        },
        'ls': {
            'lhc-orig': '-',
            'lhc-mpich3': '-',
            'lhc-openmpi3': '-',
            'lhc-mvapich2': '-',
            'sps-orig': ':',
            'sps-mpich3': ':',
            'sps-openmpi3': ':',
            'sps-mvapich2': ':',
            'ps-orig': '--',
            'ps-mpich3': '--',
            'ps-openmpi3': '--',
            'ps-mvapich2': '--',
        },
        'colors': {
            'lhc-orig': 'black',
            'lhc-mpich3': 'tab:blue',
            'lhc-openmpi3': 'tab:orange',
            'lhc-mvapich2': 'tab:green',
            'sps-orig': 'black',
            'sps-mpich3': 'tab:blue',
            'sps-openmpi3': 'tab:orange',
            'sps-mvapich2': 'tab:green',
            'ps-mpich3': 'tab:blue',
            'ps-orig': 'black',
            'ps-openmpi3': 'tab:orange',
            'ps-mvapich2': 'tab:green',
        },
        'reference': {
            'sps': {'time': 430., 'ppb': 4000000, 'turns': 100},
            'lhc': {'time': 2120., 'ppb': 2000000, 'turns': 1000},
            'ps': {'time': 1623.7, 'ppb': 4000000, 'turns': 2000},
        },

        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'x_name': 'n',
        'omp_name': 'omp',
        'y_name': 'avg_time(sec)',
        # 'y_err_name': 'std',
        'xlabel': 'Cores (x10)',
        'ylabel': 'Speedup',
        'title': 'Alternative MPI Versions',
        # 'ylim': {
        #     'speedup': [0, 210]
        # },
        # 'nticks': 6,
        'legend_loc': 'upper left',
        'figsize': (4, 4),
        'image_name': images_dir + 'mpi-versions-1.pdf'

    },

}

if __name__ == '__main__':
    for plot_key, config in plots_config.items():
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

        # print(plots_dir)
        fig = plt.figure(figsize=config['figsize'])
        # ax1 = fig.add_subplot(111)
        # ax2 = ax1.twinx()

        plt.grid(True, which='major', alpha=1)
        plt.grid(False, which='major', axis='x')
        # plt.minorticks_on()
        plt.title(config['title'])
        # ax1.set_title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        # plt.ylim(config['ylim']['speedup'])

        # plt.yscale('log', basex=2)
        # if 'ylim' in config:
        #     plt.ylim(config['ylim'])

        for key, values in plots_dir.items():
            # print(values)
            label = config['labels'][key]
            case = key.split('-')[0]

            x = np.array(values[:, header.index(config['x_name'])], float)
            omp = np.array(
                values[:, header.index(config['omp_name'])], float)
            x = (x) * omp

            y = np.array(values[:, header.index(config['y_name'])], float)
            parts = np.array(values[:, header.index('ppb')], float)
            turns = np.array(values[:, header.index('turns')], float)
            # This is the throughput
            y = parts * turns / y

            # Now the reference, 1thread
            yref = config['reference'][case]['time']
            partsref = config['reference'][case]['ppb']
            turnsref = config['reference'][case]['turns']
            yref = partsref * turnsref / yref

            speedup = y / yref

            # efficiency = 100 * speedup / x

            # We want speedup, compared to 1 worker with 1 thread
            plt.errorbar(x//10, speedup, yerr=None, color=config['colors'][key],
                         capsize=2, marker=config['markers'][key],
                         markersize=4,
                         linewidth=2., label=label,
                         ls=config['ls'][key])

            # if '10' in key:
            #     plt.xticks(x//10)
            # annotate_max(plt.gca(), x//10, speedup, ha='center', va='bottom',
            #              size='10')

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
                   labelspacing=0, borderpad=0.5, framealpha=0.8,
                   handletextpad=0.5, handlelength=2, borderaxespad=0)
        plt.tight_layout()
        save_and_crop(fig, config['image_name'], dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
