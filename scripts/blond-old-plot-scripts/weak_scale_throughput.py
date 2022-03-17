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

    'plot1': {
        'files': {
            res_dir+'raw/SPS-weak-scale-mpich3/comm-comp-report.csv': {
                'key': 'sps-mpich3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/SPS-weak-scale-mvapich2/comm-comp-report.csv': {
                'key': 'sps-mvapich2',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/SPS-weak-scale-openmpi3/comm-comp-report.csv': {
                'key': 'sps-openmpi3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/LHC-weak-scale-mpich3/comm-comp-report.csv': {
                'key': 'lhc-mpich3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/LHC-weak-scale-openmpi3/comm-comp-report.csv': {
                'key': 'lhc-openmpi3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/LHC-weak-scale-mvapich2/comm-comp-report.csv': {
                'key': 'lhc-mvapich2',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/PS-weak-scale-mpich3/comm-comp-report.csv': {
                'key': 'ps-mpich3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/PS-weak-scale-openmpi3/comm-comp-report.csv': {
                'key': 'ps-openmpi3',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
            res_dir+'raw/PS-weak-scale-mvapich2/comm-comp-report.csv': {
                'key': 'ps-mvapich2',
                'lines': {
                    'omp': ['10'],
                    'type': ['total']
                }
            },
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
            'lhc': 'o',
            'sps': 's',
            'ps': 'x'
            # 'lhc-mpich3': 'o',
            # 'lhc-orig': 'o',
            # 'lhc-openmpi3': 'o',
            # 'lhc-mvapich2': 'o',
            # 'sps-mpich3': 's',
            # 'sps-orig': 's',
            # 'sps-openmpi3': 's',
            # 'sps-mvapich2': 's',
            # 'ps-orig': 'x',
            # 'ps-mpich3': 'x',
            # 'ps-openmpi3': 'x',
            # 'ps-mvapich2': 'x',
        },
        'ls': {
            'lhc': '-',
            'sps': ':',
            'ps': '--'
            # 'lhc-orig': '-',
            # 'lhc-mpich3': '-',
            # 'lhc-openmpi3': '-',
            # 'lhc-mvapich2': '-',
            # 'sps-orig': ':',
            # 'sps-mpich3': ':',
            # 'sps-openmpi3': ':',
            # 'sps-mvapich2': ':',
            # 'ps-orig': '--',
            # 'ps-mpich3': '--',
            # 'ps-openmpi3': '--',
            # 'ps-mvapich2': '--',
        },
        'colors': {
            # 'orig': 'black',
            'mpich3': 'tab:blue',
            'openmpi3': 'tab:orange',
            'mvapich2': 'tab:green',
            # 'sps-orig': 'black',
            # 'sps-mpich3': 'tab:blue',
            # 'sps-openmpi3': 'tab:orange',
            # 'sps-mvapich2': 'tab:green',
            # 'ps-mpich3': 'tab:blue',
            # 'ps-orig': 'black',
            # 'ps-openmpi3': 'tab:orange',
            # 'ps-mvapich2': 'tab:green',
        },
        'reference': {
            'sps': {'time': 430., 'parts': 4000000, 'turns': 100},
            'lhc': {'time': 2120., 'parts': 2000000, 'turns': 1000},
            'ps': {'time': 1623.7, 'parts': 4000000, 'turns': 2000},
        },

        # 'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
        'x_name': 'n',
        'omp_name': 'omp',
        'y_name': 'avg_time(sec)',
        # 'y_err_name': 'std',
        'xlabel': 'Cores (x10)',
        'ylabel': 'Speedup',
        'title': 'Weak scale, alternative MPI Versions',
        # 'ylim': {
        #     'speedup': [0, 210]
        # },
        # 'nticks': 6,
        'legend_loc': 'upper left',
        'figsize': (4, 4),
        'image_name': {
            'norm_time': images_dir + 'mpi-vers-ws-norm_time-1.pdf',
            'throughput': images_dir + 'mpi-vers-ws-throughput-1.pdf'
        }

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
        for metric in ['norm_time', 'throughput']:
            fig = plt.figure(figsize=config['figsize'])

            plt.grid(True, which='major', alpha=1)
            plt.grid(False, which='major', axis='x')
            # plt.minorticks_on()
            plt.title(config['title'])
            # ax1.set_title(config['title'])
            plt.xlabel(config['xlabel'])
            plt.ylabel(metric)
            # plt.ylim(config['ylim']['speedup'])

            # plt.yscale('log', basex=2)
            # if 'ylim' in config:
            #     plt.ylim(config['ylim'])

            for key, values in plots_dir.items():
                # print(values)
                label = config['labels'][key]
                case = key.split('-')[0]
                mpiv = key.split('-')[1]

                x = get_values(values, header, config['x_name'])
                omp = get_values(values, header, config['omp_name'])
                x = x * omp

                y = get_values(values, header, config['y_name'])
                parts = get_values(values, header, 'parts')
                bunches = get_values(values, header, 'bunches')
                turns = get_values(values, header, 'turns')
                if metric == 'throughput':
                    y = parts * bunches * turns / y
                sortid = np.argsort(x)
                x = x[sortid]
                y = y[sortid]
                # if metric == 'norm_time':
                y = y / y[0]

                # # Now the reference, 1thread
                # yref = config['reference'][case]['time']
                # partsref = config['reference'][case]['parts']
                # turnsref = config['reference'][case]['turns']
                # yref = partsref * turnsref / yref

                # speedup = y / yref

                # efficiency = 100 * speedup / x

                # We want speedup, compared to 1 worker with 1 thread
                plt.errorbar(x//10, y, yerr=None,
                             color=config['colors'][mpiv],
                             marker=config['markers'][case],
                             capsize=2, markersize=4, linewidth=1.,
                             # label=label,
                             ls=config['ls'][case])

                # if '10' in key:
                #     plt.xticks(x//10)
                # annotate_max(plt.gca(), x//10, speedup, ha='center', va='bottom',
                #              size='10')

                # ax2.errorbar(x//10, efficiency, yerr=None, color=config['colors']['efficiency'],
                #              capsize=2, marker=config['markers'][key], markersize=4,
                #              linewidth=1.)

            plt.xticks(x//10)
            handles = []
            for k, v in config['markers'].items():
                line = plt.plot([], [], color='black',
                                     ls=config['ls'][k],
                                     marker=v, label=k)
                handles.append(line)
            for k, v in config['colors'].items():
                plt.plot([], [], color=v, ls='-', label=k,
                    lw=4)


            plt.legend(loc=config['legend_loc'], fancybox=True, fontsize=10.5,
                       labelspacing=0, borderpad=0.5, framealpha=0.8,
                       handletextpad=0.5, handlelength=2, borderaxespad=0)
            plt.tight_layout()
            save_and_crop(fig, config['image_name']
                          [metric], dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()
