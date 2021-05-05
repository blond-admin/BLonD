import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import h5py
import argparse
import sys
import os
import matplotlib.lines as mlines
from plot.plotting_utilities import *
from scipy import stats
from cycler import cycle
import bisect


this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(
    description='Evaluate single precision raw data.')

parser.add_argument('-i', '--indir', type=str,
                    help='Directory with input files.')

parser.add_argument('-c', '--cases', nargs='+', choices=['lhc', 'sps', 'ps'],
                    default=['lhc', 'sps', 'ps'],
                    help='Which testcases to plot.')

parser.add_argument('-t', '--techniques', nargs='+',
                    choices=['base', 'rds', 'srp', 'f32', 'f32-srp', 'f32-rds'],
                    default=['base', 'f32', 'rds', 'f32-rds', 'srp', 'f32-srp'],
                    help='which techniqeus to plot')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='Directory to store the results.')

# parser.add_argument('-ymin', '--ymin', type=float, default=None,
#                     help='Min value for y axis.')

# parser.add_argument('-ymax', '--ymax', type=float, default=None,
#                     help='Max value for y axis.')

# parser.add_argument('-reduce', '--reduce', type=int, default=[], nargs='+',
#                     help='Plot lines for these reduce intervals. \n' +
#                     'Default: Use all the available')

parser.add_argument('-last', '--last', type=int, default=None,
                    help='Last turn to consider. default: None (last).')

parser.add_argument('-first', '--first', type=int, default=0,
                    help='First turn to consider. default: 0.')


parser.add_argument('-p', '--points', type=int, default=-1,
                    help='Num of points in the plot. Default: all')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plot or save only. Default: save only')


gconfig = {
    'hatches': ['', '', 'xx'],
    'markers': ['x', 'o', '^'],
    'boxcolor': 'xkcd:light blue',
    'linecolor': 'xkcd:blue',
    'colors': ['xkcd:red', 'xkcd:green', 'xkcd:blue'],
    'group': '/default',
    'labels': {'std_profile': r'$s_{profile}$',
               'std_dE': r'$s_{dE}$',
               'std_dt': r'$s_{dt}$'},
    'x_name': 'turns',
    'y_names': [
        'std_profile',
        'std_dE',
        'std_dt',
        # 'mean_profile',
        # 'mean_dE',
        # 'mean_dt'
    ],
    # 'y_err_name': 'std',
    'xlabel': {'xlabel': 'Turn', 'labelpad': 1, 'fontsize': 10},
    'ylabel': {'ylabel': r'Relative Error', 'labelpad': 1, 'fontsize': 10},
    'title': {
        # 's': '{}'.format(case.upper()),
        'fontsize': 10,
        # 'y': .95,
        # 'x': 0.45,
        'fontweight': 'bold',
    },
    'figsize': [4, 3],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 10},
    'fontsize': 10,
    'legend': {
        'loc': 'best', 'ncol': 3, 'handlelength': 1.5, 'fancybox': False,
        'framealpha': .7, 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        # 'bbox_to_anchor': (0., 1.05)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'boxplot': {
        'notch': False,
        'vert': False,
        'patch_artist': True,
        'showcaps': True,
        'showbox': True,
        'showfliers': False,

        'capprops': None,
        'boxprops': None,
        'whiskerprops': None,
        'flierprops': None,
        'medianprops': None,
    },
    'fontname': 'DejaVu Sans Mono',
    # 'err_formula': 'np.sqrt((1 - inputd["std_dt"] / based["std_dt"])**2 + (1 - inputd["std_dE"] / based["std_dE"])**2)',
    # 'err_formula': 'np.sqrt(0*(1 - inputd["std_dt"] / based["std_dt"])**2 + (1 - inputd["std_dE"] / based["std_dE"])**2)',
    # 'err_formula': 'np.sqrt((1 - inputd["losses"] / based["losses"])**2)',
    # 'err_formula': 'np.sqrt((1 - inputd["mean_dt"] / based["mean_dt"])**2 + (1 - inputd["mean_dE"] / based["mean_dE"])**2)',
    # 'err_formula': 'np.sqrt(((based["mean_dt"] - inputd["mean_dt"]) / np.maximum(inputd["mean_dt"], based["mean_dt"]))**2 + ((based["mean_dE"] - inputd["mean_dE"]) / np.maximum(inputd["mean_dE"], based["mean_dE"]))**2 + ((based["std_dt"] - inputd["std_dt"]) / np.maximum(inputd["std_dt"], based["std_dt"]))**2 + ((based["std_dE"] - inputd["std_dE"]) / np.maximum(inputd["std_dE"], based["std_dE"]))**2)',
    'err_formula': 'np.sqrt(((based["mean_dt"] - inputd["mean_dt"]) / np.maximum(inputd["mean_dt"], based["mean_dt"]))**2 + 0*((based["mean_dE"] - inputd["mean_dE"]) / np.maximum(inputd["mean_dE"], based["mean_dE"]))**2 + ((based["std_dt"] - inputd["std_dt"]) / np.maximum(inputd["std_dt"], based["std_dt"]))**2 + 0*((based["std_dE"] - inputd["std_dE"]) / np.maximum(inputd["std_dE"], based["std_dE"]))**2)',
    # 'err_formula': 'np.sqrt((based["mean_dt"] - inputd["mean_dt"] / based["mean_dt"])**2 + (1 - inputd["std_dt"] / based["std_dt"])**2 + (1 - inputd["mean_dE"] / based["mean_dE"])**2 + (1 - inputd["std_dE"] / based["std_dE"])**2)',
    # 'ylim': [-7, -2],
    'ylim': {
        'lhc': [-2, 1],
        'ps': [-5, 1],
        'sps': [-2, 1],
    },
    # 'xlim': [1.6, 36],
    # 'yticks': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'outfiles': ['{}/{}-{}.png',
                 # '{}/{}-{}.pdf'
                 ],
    'plots': {
        'ps-f32': {
            'in': {  # 'f32': '{}/{}/float32/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/27Aug20.16-13-20-47/monitor.h5',
                'f32': '{}/{}/float32/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_partition_default/19Apr21.11-29-34-25/monitor.h5',
            },
            'title': 'PS F32 40kT 1x4M P',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed2_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-12-47-99/monitor.h5',
            'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-40-40-8/monitor.h5',
        },
        'ps-srp': {
            'in': {  # 'r2': '{}/{}/srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red2_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-45-02-45/monitor.h5',
                # 'r3': '{}/{}/srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red3_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-56-17-69/monitor.h5',
                'r3': '{}/{}/srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red3_mtw50_seed0_approx1_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-02-22-56/monitor.h5',
                # 'r4': '{}/{}/srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red4_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-07-19-15/monitor.h5'
            },
            'title': 'PS SRP 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-40-40-8/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed2_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-12-47-99/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
            # 'base': '{}/{}/exact/_p1000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-54-01-69/monitor.h5',
        },
        'ps-rds': {
            'in': {  # 'w2': '{}/{}/rds-monitor/_p4000000_b1_s256_t40000_w2_o4_N1_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-17-38-35/monitor.h5',
                # 'w4': '{}/{}/rds-monitor/_p4000000_b1_s256_t40000_w4_o4_N1_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-27-50-87/monitor.h5',
                'w4': '{}/{}/rds-monitor/_p4000000_b1_s256_t40000_w4_o1_N1_red1_mtw50_seed0_approx2_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-25-34-41/monitor.h5'
                # 'w8': '{}/{}/rds-monitor/_p4000000_b1_s256_t40000_w8_o4_N2_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-38-59-37/monitor.h5'
            },
            'title': 'PS RDS 40kT 1x4M P',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed2_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-12-47-99/monitor.h5',
            'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-40-40-8/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
            # 'base': '{}/{}/exact/_p1000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-54-01-69/monitor.h5',
        },
        'ps-f32-srp': {
            'in': {  # 'r2': '{}/{}/f32-srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red2_mtw50_seed0_approx1_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-07-02-83/monitor.h5',
                'r3': '{}/{}/f32-srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red3_mtw50_seed0_approx1_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_partition_default/19Apr21.11-42-22-71/monitor.h5',
                # 'r3': '{}/{}/f32-srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red3_mtw50_seed0_approx1_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-17-10-47/monitor.h5',
                # 'r4': '{}/{}/f32-srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red4_mtw50_seed0_approx1_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-27-25-20/monitor.h5'
            },
            'title': 'PS F32-SRP 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-40-40-8/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
        },
        'ps-f32-rds': {
            'in': {  # 'w2': '{}/{}/f32-rds-monitor/_p4000000_b1_s256_t40000_w2_o4_N1_red1_mtw50_seed0_approx2_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-36-26-7/monitor.h5',
                # 'w4': '{}/{}/f32-rds-monitor/_p4000000_b1_s256_t40000_w4_o4_N1_red1_mtw50_seed0_approx2_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-44-50-76/monitor.h5',
                'w4': '{}/{}/f32-rds-monitor/_p4000000_b1_s256_t40000_w4_o1_N1_red1_mtw50_seed0_approx2_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_partition_default/19Apr21.11-51-04-71/monitor.h5'
                # 'w8': '{}/{}/f32-rds-monitor/_p4000000_b1_s256_t40000_w8_o4_N2_red1_mtw50_seed0_approx2_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-52-03-4/monitor.h5'
            },
            'title': 'PS F32-RDS 40kT 1x4M P',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed2_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-12-47-99/monitor.h5',
            'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-40-40-8/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
        },
        'ps-base': {
            'in': {
                # 'seed0-': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-40-40-8/monitor.h5',
                # 'seed1-': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed1_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-56-27-16/monitor.h5',
                'seed2-': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed2_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-12-47-99/monitor.h5',

                # 'seed1-': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-32-35-6/monitor.h5',
                # 'seed2-': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-44-08-46/monitor.h5',
                # 'seed3-': 'results/precision-analysis/ps/precision-seed/_p1000000_b1_s256_t40000_w1_o14_N1_red1_mtw50_seed3_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precdouble_/29Apr20.13-33-05-20/monitor.h5',
            },
            'title': 'PS 40kT seeds 1x4M P',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed2_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-12-47-99/monitor.h5',
            'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-40-40-8/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
        },



        'sps-f32': {
            'in': {  # 'f32': '{}/{}/float32/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/27Aug20.16-13-09-11/monitor.h5',
                'f32': '{}/{}/float32/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_partition_default/19Apr21.12-27-21-84/monitor.h5',
            },
            'title': 'SPS F32 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-06-54-84/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
        },
        'sps-srp': {
            'in': {  # 'r2': '{}/{}/srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red2_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-54-44-38/monitor.h5',
                'r3': '{}/{}/srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-01-37-73/monitor.h5',
                # 'r3': '{}/{}/srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-20-10-24/monitor.h5',
                # 'r4': '{}/{}/srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red4_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-41-26-15/monitor.h5'
            },
            'title': 'SPS SRP 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-06-54-84/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
        },
        'sps-rds': {
            'in': {  # 'w2': '{}/{}/rds-monitor/_p4000000_b1_s1408_t40000_w2_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.16-01-36-53/monitor.h5',
                # 'w4': '{}/{}/rds-monitor/_p4000000_b1_s1408_t40000_w4_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.16-30-49-28/monitor.h5',
                'w4': '{}/{}/rds-monitor/_p4000000_b1_s1408_t40000_w4_o1_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-29-23-17/monitor.h5',
                # 'w8': '{}/{}/rds-monitor/_p4000000_b1_s1408_t40000_w8_o4_N2_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.16-57-27-55/monitor.h5'
            },
            'title': 'SPS RDS 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-06-54-84/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
            # 'base': '{}/{}/exact/_p1000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-52-48-93/monitor.h5',
        },
        'sps-f32-srp': {
            'in': {  # 'r2': '{}/{}/f32-srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red2_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-06-58-33/monitor.h5',
                'r3': '{}/{}/f32-srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_partition_default/19Apr21.12-33-21-91/monitor.h5',
                # 'r3': '{}/{}/f32-srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-25-41-5/monitor.h5',
                # 'r4': '{}/{}/f32-srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red4_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-41-14-8/monitor.h5'
            },
            'title': 'SPS F32-SRP 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-06-54-84/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
        },
        'sps-f32-rds': {
            'in': {  # 'w2': '{}/{}/f32-rds-monitor/_p4000000_b1_s1408_t40000_w2_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-54-53-17/monitor.h5',
                # 'w4': '{}/{}/f32-rds-monitor/_p4000000_b1_s1408_t40000_w4_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.13-13-23-79/monitor.h5',
                'w4': '{}/{}/f32-rds-monitor/_p4000000_b1_s1408_t40000_w4_o1_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_partition_default/19Apr21.12-34-02-69/monitor.h5',
                # 'w8': '{}/{}/f32-rds-monitor/_p4000000_b1_s1408_t40000_w8_o4_N2_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.13-29-13-91/monitor.h5'
            },
            'title': 'SPS F32-RDS 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-06-54-84/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
            # 'base': '{}/{}/exact/_p1000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-52-48-93/monitor.h5',
        },
        'sps-base': {
            'in': {
                'seed1-': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-37-35-77/monitor.h5',
                'seed2-': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.12-04-18-95/monitor.h5',

                # 'seed1-': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.18-00-58-23/monitor.h5',
                # 'seed2-': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.18-27-07-24/monitor.h5',
                # 'seed3-': '{}/{}/exact/_p1000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed3_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.13-42-23-86/monitor.h5',
            },
            'title': 'SPS 40kT seeds 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-06-54-84/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
            # 'base': '{}/{}/exact/_p1000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-52-48-93/monitor.h5',
        },


        'lhc-f32': {
            'in': {  # 'f32': '{}/{}/float32/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/27Aug20.16-13-01-28/monitor.h5',
                'f32': '{}/{}/float32/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_partition_default/19Apr21.11-29-24-80/monitor.h5',
                # 'f32': '{}/{}/float32/_p10000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_/15Apr21.22-35-41-72/monitor.h5',
            },
            'title': 'LHC F32 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-46-43-74/monitor.h5',
            # 'base': '{}/{}/exact/_p10000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_/15Apr21.20-46-51-93/monitor.h5',
        },
        'lhc-f32-srp': {
            'in': {  # 'r2': '{}/{}/f32-srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red2_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-25-20-38/monitor.h5',
                # 'r3': '{}/{}/f32-srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-33-56-37/monitor.h5',
                # 'r3': '{}/{}/f32-srp-monitor/_p10000000_b1_s1000_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_/15Apr21.23-06-27-29/monitor.h5',
                'r3': '{}/{}/f32-srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_partition_default/19Apr21.11-40-21-57/monitor.h5',
                # 'r4': '{}/{}/f32-srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red4_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-41-38-83/monitor.h5'
            },
            'title': 'LHC F32-SRP 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-46-43-74/monitor.h5',
            # 'base': '{}/{}/exact/_p10000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_/15Apr21.20-46-51-93/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
        },
        'lhc-f32-rds': {
            'in': {  # 'w2': '{}/{}/f32-rds-monitor/_p4000000_b1_s1000_t40000_w2_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-06-43-76/monitor.h5',
                # 'w4': '{}/{}/f32-rds-monitor/_p4000000_b1_s1000_t40000_w4_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-13-30-88/monitor.h5',
                'w4': '{}/{}/f32-rds-monitor/_p4000000_b1_s1000_t40000_w4_o1_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_gpu0_partition_default/19Apr21.11-47-47-42/monitor.h5'
                # 'w8': '{}/{}/f32-rds-monitor/_p4000000_b1_s1000_t40000_w8_o4_N2_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-19-16-89/monitor.h5'
            },
            'title': 'LHC F32-RDS 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-46-43-74/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
        },
        'lhc-srp': {
            'in': {  # 'r2': '{}/{}/srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red2_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-44-53-65/monitor.h5',
                # 'r2': '{}/{}/srp-monitor/_p10000000_b1_s1000_t40000_w1_o4_N1_red2_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_/15Apr21.17-29-49-59/monitor.h5',
                # 'r3': '{}/{}/srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-54-47-86/monitor.h5',
                # 'r3': '{}/{}/srp-monitor/_p10000000_b1_s1000_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_/15Apr21.20-13-20-27/monitor.h5',
                'r3': '{}/{}/srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-25-38-88/monitor.h5',
                # 'r4': '{}/{}/srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red4_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-05-03-50/monitor.h5'
                # 'r4': '{}/{}/srp-monitor/_p10000000_b1_s1000_t40000_w1_o4_N1_red4_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_/15Apr21.18-25-06-6/monitor.h5'
            },
            'title': 'LHC SRP 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-46-43-74/monitor.h5',
            # 'base': '{}/{}/exact/_p10000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_/15Apr21.20-46-51-93/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
        },
        'lhc-rds': {
            'in': {  # 'w2': '{}/{}/rds-monitor/_p4000000_b1_s1000_t40000_w2_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-15-04-0/monitor.h5',
                # 'w4': '{}/{}/rds-monitor/_p4000000_b1_s1000_t40000_w4_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-23-15-25/monitor.h5',
                'w4': '{}/{}/rds-monitor/_p4000000_b1_s1000_t40000_w4_o1_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-34-53-1/monitor.h5'
                # 'w8': '{}/{}/rds-monitor/_p4000000_b1_s1000_t40000_w8_o4_N2_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-30-38-48/monitor.h5'
            },
            'title': 'LHC RDS 40kT 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-46-43-74/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
            # 'base': '{}/{}/exact/_p1000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-53-16-60/monitor.h5',
        },
        'lhc-base': {
            'in': {
                # 'seed1-': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-55-09-57/monitor.h5',
                # 'seed2-': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.16-09-29-31/monitor.h5',
                # 'seed1-': '{}/{}/exact/_p10000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_/15Apr21.21-23-27-46/monitor.h5',
                # 'seed2-': '{}/{}/exact/_p10000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_/15Apr21.21-59-36-3/monitor.h5',
                'seed1-': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-00-45-38/monitor.h5',
                'seed2-': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.11-14-31-95/monitor.h5',

            },
            'title': 'LHC 40kT seeds 1x4M P',
            'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_partition_default/19Apr21.10-46-43-74/monitor.h5',
            # 'base': '{}/{}/exact/_p10000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_gpu0_/15Apr21.20-46-51-93/monitor.h5',
            # 'base': '{}/{}/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
        },

    },

}

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
# Force sans-serif math mode (for axes labels)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
plt.rcParams['font.family'] = 'sans-serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica'


if __name__ == '__main__':

    args = parser.parse_args()

    first_t = args.first
    last_t = args.last
    outdir = args.outdir
    points = args.points
    indir = args.indir
    assert indir, 'You must provide the indir argument'

    res_dir = args.outdir
    images_dir = os.path.join(res_dir)

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # cases = args.cases
    # techniques = args.techniques

    # There will be one plot per testcase
    for case in args.cases:
        fig, ax = plt.subplots(ncols=1, nrows=1,
                               sharex=True, sharey=True,
                               figsize=gconfig['figsize'])
        pos = 0
        step = 1
        width = .7 * step
        xtickspos = []
        xticks = []
        # I need to collect the data
        # It's best if I collect, plot, then continue
        for tech in args.techniques:
            inkey = f'{case}-{tech}'
            infiles = gconfig['plots'][inkey]['in']
            for infk, infv in infiles.items():
                infiles[infk] = infv.format(indir, case)
            basefile = gconfig['plots'][inkey]['base']
            basefile = basefile.format(indir, case)

            # Read basefile
            tempd = {}
            fullfile = basefile
            h5file = h5py.File(fullfile, 'r')
            for key in h5file[gconfig['group']]:
                val = h5file[gconfig['group']][key][()]
                if key not in tempd:
                    # tempd[key] = val.reshape(len(val))
                    tempd[key] = val.flatten()
                tempd[key] = tempd[key][first_t:]
                if last_t:
                    tempd[key] = tempd[key][:last_t]
            h5file.close()
            # del tempd[gconfig['x_name']]
            based = tempd.copy()
            turns = based[gconfig['x_name']]

            tempd = {}

            # Read infile
            for keyf, infile in infiles.items():
                fullfile = infile
                h5file = h5py.File(fullfile, 'r')
                if keyf not in tempd:
                    tempd[keyf] = {}
                for key in h5file[gconfig['group']]:
                    val = h5file[gconfig['group']][key][()]
                    if key not in tempd:
                        # tempd[keyf][key] = val.reshape(len(val))
                        tempd[keyf][key] = val.flatten()
                    tempd[keyf][key] = tempd[keyf][key][first_t:]
                    if last_t:
                        tempd[keyf][key] = tempd[keyf][key][:last_t]

                h5file.close()
                assert np.array_equal(turns, tempd[keyf][gconfig['x_name']])
                # del tempd[keyf][gconfig['x_name']]

            ind = tempd.copy()

            points = min(len(turns), points) if points > 0 else len(turns)
            intv = int(np.ceil(len(turns)/points))

            errordist = []
            for keyf, inputd in ind.items():
                # inputd = ind[kef]
                try:
                    errors = eval(gconfig['err_formula'])
                except Exception as e:
                    print(e)
                    continue
                errors = errors[::intv]
                errors = sorted(errors)
                errors = errors[int(.10 * len(errors)):int(.9 * len(errors))]
                errordist += list(errors)
                # errordist = list(errors)
            errordist = np.log10(errordist)
            bplot = plt.boxplot(errordist, widths=width, positions=[
                                pos], **gconfig['boxplot'])

            print('{} Median: {:e}'.format(inkey, np.median(errordist)))
            for patch in bplot['boxes']:
                patch.set_facecolor(gconfig['boxcolor'])
                patch.set_edgecolor(gconfig['linecolor'])
            for patch in bplot['medians']:
                patch.set_color(gconfig['linecolor'])
            for patch in bplot['caps']:
                patch.set_color(gconfig['linecolor'])
            for patch in bplot['whiskers']:
                patch.set_color(gconfig['linecolor'])

            xtickspos.append(pos)
            xticks.append(tech)
            pos += step
        # plt.yscale('log')
        plt.grid(True, which='both', axis='both', alpha=0.5)
        # plt.grid(False, which='major', axis='x')
        plt.title(case.upper(), **gconfig['title'])
        # plt.xlabel(**gconfig['xlabel'])
        # plt.ylabel(**gconfig['ylabel'])
        if gconfig['boxplot']['vert'] == True:

            plt.ylim(gconfig['ylim'])
            plt.xticks(xtickspos, xticks, **gconfig['ticks'])
            yticks = ['{:1.0e}'.format(10**i) for i in range(gconfig['ylim'][0],
                                                             gconfig['ylim'][1]+1)]
            plt.yticks(np.arange(gconfig['ylim'][0],
                                 gconfig['ylim'][1]+1), yticks, **gconfig['ticks'])
        else:
            plt.yticks(xtickspos, xticks, **gconfig['ticks'])
            ylims = gconfig['ylim'][case]
            yticks = ['{:1.0e}'.format(10**i) for i in range(ylims[0], ylims[1]+1)]
            plt.xlim(ylims)
            plt.xticks(np.arange(ylims[0], ylims[1]+1),
                       yticks, **gconfig['ticks'])

        ax.tick_params(**gconfig['tick_params'])
        plt.tight_layout()
        # plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir, this_filename[: -3], inkey)
            print('[{}] {}: {}'.format(
                this_filename[: -3], 'Saving figure', file))

            save_and_crop(fig, file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
