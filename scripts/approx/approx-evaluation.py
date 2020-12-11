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
    description='Evaluate the approximations raw data.')

# parser.add_argument('-i', '--infile', type=str, default=None,
#                     help='Input .h5 file.')

# parser.add_argument('-b', '--basefile', type=str, default=None,
#                     help='Base .h5 files.')

# parser.add_argument('-i', '--inputkey', type=str, default='2kT-acc',
#                     choices=['2kT-acc', '1mT-acc', '1mT-noacc',
#                              '1mT-acc-seed'],
#                     help='Key of the input config.')


parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='Directory to store the results.')

# parser.add_argument('-ymin', '--ymin', type=float, default=None,
#                     help='Min value for y axis.')

# parser.add_argument('-ymax', '--ymax', type=float, default=None,
#                     help='Max value for y axis.')

# parser.add_argument('-reduce', '--reduce', type=int, default=[], nargs='+',
#                     help='Plot lines for these reduce intervals. \n' +
#                     'Default: Use all the available')

parser.add_argument('-turns', '--turns', type=int, default=None,
                    help='Last turn to plot (default: plot all the turns).')

parser.add_argument('-p', '--points', type=int, default=-1,
                    help='Num of points in the plot. Default: all')

# parser.add_argument('-t', '--ts', type=str, default=['1'], nargs='+',
# help='Running mean window. Default: [1]')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plot or save only. Default: save only')


args = parser.parse_args()

res_dir = args.outdir
images_dir = os.path.join(res_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)


gconfig = {
    'hatches': ['', '', 'xx'],
    'markers': ['x', 'o', '^'],
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
    'xlabel': {'xlabel': 'Turn', 'labelpad': 1, 'fontsize': 9},
    'ylabel': {'ylabel': r'Relative Error', 'labelpad': 1, 'fontsize': 9},
    'title': {
        # 's': '{}'.format(case.upper()),
        'fontsize': 9,
        'y': .95,
        # 'x': 0.45,
        'fontweight': 'bold',
    },
    'figsize': [5.4, 1.6],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 9},
    'fontsize': 9,
    'legend': {
        'loc': 'upper left', 'ncol': 3, 'handlelength': 1., 'fancybox': False,
        'framealpha': .7, 'fontsize': 9, 'labelspacing': 0, 'borderpad': 0.2,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.4,
        # 'bbox_to_anchor': (0., 1.05)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    'ylim': [1e-8, 1],
    # 'xlim': [1.6, 36],
    # 'yticks': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'outfiles': ['{}/{}-{}.png', '{}/{}-{}.pdf'],
    # 'cases': ['ex01'],
    'inputkeys': [
        'psb-f32',
        'psb-srp',
        'psb-rds',
        'psb-f32-srp',
        'psb-f32-rds',
        'psb-40kt-seed',
        # 'ps-f32',
        # 'ps-srp',
        # 'ps-rds',
        'ps-f32-srp',
        'ps-f32-rds',

        # 'ps-40kt-seed',
        # 'sps-f32',
        # 'sps-srp',
        # 'sps-rds',
        'sps-f32-srp',
        'sps-f32-rds',

        # 'sps-40kt-seed',
        # 'lhc-f32',
        # 'lhc-srp',
        # 'lhc-rds',
        'lhc-f32-srp',
        'lhc-f32-rds',

        # 'lhc-40kt-seed',
    ],
    'plots': {
        'psb-f32': {
            'in': {'f32': 'results/approx-eval/psb/float32/_p4000000_b1_s128_t35000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.13-57-46-36/monitor.h5',
                   },
            'title': 'PSB F32 35kT 1x4M P',
            'base': 'results/approx-eval/psb/exact/_p4000000_b1_s128_t35000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-12-15-77/monitor.h5',
        },
        'psb-f32-srp': {
            'in': {'r2': 'results/approx-eval/psb/f32-srp-monitor/_p4000000_b1_s128_t35000_w1_o4_N1_red2_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.14-10-48-86/monitor.h5',
                   'r3': 'results/approx-eval/psb/f32-srp-monitor/_p4000000_b1_s128_t35000_w1_o4_N1_red3_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.14-16-06-25/monitor.h5',
                   'r4': 'results/approx-eval/psb/f32-srp-monitor/_p4000000_b1_s128_t35000_w1_o4_N1_red4_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.14-21-09-30/monitor.h5'
                   },
            'title': 'PSB F32-SRP 35kT 1x4M P',
            'base': 'results/approx-eval/psb/exact/_p4000000_b1_s128_t35000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-12-15-77/monitor.h5',
        },
        'psb-f32-rds': {
            'in': {'w2': 'results/approx-eval/psb/f32-rds-monitor/_p4000000_b1_s128_t35000_w2_o4_N1_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.14-03-51-87/monitor.h5',
                   'w4': 'results/approx-eval/psb/f32-rds-monitor/_p4000000_b1_s128_t35000_w4_o4_N1_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.14-07-15-34/monitor.h5',
                   'w8': 'results/approx-eval/psb/f32-rds-monitor/_p4000000_b1_s128_t35000_w8_o4_N2_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.14-09-11-18/monitor.h5'
                   },
            'title': 'PSB F32-RDS 35kT 1x4M P',
            'base': 'results/approx-eval/psb/exact/_p4000000_b1_s128_t35000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-12-15-77/monitor.h5',
        },
        'psb-srp': {
            'in': {'r2': 'results/approx-eval/psb/srp-monitor/_p4000000_b1_s128_t35000_w1_o4_N1_red2_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.12-06-54-52/monitor.h5',
                   'r3': 'results/approx-eval/psb/srp-monitor/_p4000000_b1_s128_t35000_w1_o4_N1_red3_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.12-24-21-99/monitor.h5',
                   'r4': 'results/approx-eval/psb/srp-monitor/_p4000000_b1_s128_t35000_w1_o4_N1_red4_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.12-41-48-65/monitor.h5'
                   },
            'title': 'PSB SRP 35kT 1x4M P',
            'base': 'results/approx-eval/psb/exact/_p4000000_b1_s128_t35000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-12-15-77/monitor.h5',
        },
        'psb-rds': {
            'in': {'w2': 'results/approx-eval/psb/rds-monitor/_p4000000_b1_s128_t35000_w2_o4_N1_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.12-58-23-37/monitor.h5',
                   'w4': 'results/approx-eval/psb/rds-monitor/_p4000000_b1_s128_t35000_w4_o4_N1_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-05-15-93/monitor.h5',
                   'w8': 'results/approx-eval/psb/rds-monitor/_p4000000_b1_s128_t35000_w8_o4_N2_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-09-17-18/monitor.h5'
                   },
            'title': 'PSB RDS 35kT 1x4M P',
            'base': 'results/approx-eval/psb/exact/_p4000000_b1_s128_t35000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-12-15-77/monitor.h5',
        },
        'psb-40kt-seed': {
            'in': {
                'seed1-': 'results/approx-eval/psb/exact/_p4000000_b1_s128_t35000_w1_o4_N1_red1_mtw50_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-26-13-16/monitor.h5',
                'seed2-': 'results/approx-eval/psb/exact/_p4000000_b1_s128_t35000_w1_o4_N1_red1_mtw50_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-43-02-16/monitor.h5',
            },
            'title': 'PSB 35kT seeds 1x4M P',
            'base': 'results/approx-eval/psb/exact/_p4000000_b1_s128_t35000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/28Sep20.13-12-15-77/monitor.h5',
        },
        'ps-f32': {
            'in': {'f32': 'results/approx-eval/ps/float32/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/27Aug20.16-13-20-47/monitor.h5',
                   },
            'title': 'PS F32 40kT 1x4M P',
            'base': 'results/approx-eval/ps/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
        },
        'ps-srp': {
            'in': {'r2': 'results/approx-eval/ps/srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red2_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-45-02-45/monitor.h5',
                   'r3': 'results/approx-eval/ps/srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red3_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-56-17-69/monitor.h5',
                   'r4': 'results/approx-eval/ps/srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red4_mtw50_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-07-19-15/monitor.h5'
                   },
            'title': 'PS SRP 40kT 1x4M P',
            'base': 'results/approx-eval/ps/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
            # 'base': 'results/approx-eval/ps/exact/_p1000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-54-01-69/monitor.h5',
        },
        'ps-rds': {
            'in': {'w2': 'results/approx-eval/ps/rds-monitor/_p4000000_b1_s256_t40000_w2_o4_N1_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-17-38-35/monitor.h5',
                   'w4': 'results/approx-eval/ps/rds-monitor/_p4000000_b1_s256_t40000_w4_o4_N1_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-27-50-87/monitor.h5',
                   'w8': 'results/approx-eval/ps/rds-monitor/_p4000000_b1_s256_t40000_w8_o4_N2_red1_mtw50_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-38-59-37/monitor.h5'
                   },
            'title': 'PS RDS 40kT 1x4M P',
            'base': 'results/approx-eval/ps/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
            # 'base': 'results/approx-eval/ps/exact/_p1000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-54-01-69/monitor.h5',
        },
        'ps-f32-srp': {
            'in': {'r2': 'results/approx-eval/ps/f32-srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red2_mtw50_seed0_approx1_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-07-02-83/monitor.h5',
                   'r3': 'results/approx-eval/ps/f32-srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red3_mtw50_seed0_approx1_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-17-10-47/monitor.h5',
                   'r4': 'results/approx-eval/ps/f32-srp-monitor/_p4000000_b1_s256_t40000_w1_o4_N1_red4_mtw50_seed0_approx1_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-27-25-20/monitor.h5'
                   },
            'title': 'PS F32-SRP 40kT 1x4M P',
            'base': 'results/approx-eval/ps/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
        },
        'ps-f32-rds': {
            'in': {'w2': 'results/approx-eval/ps/f32-rds-monitor/_p4000000_b1_s256_t40000_w2_o4_N1_red1_mtw50_seed0_approx2_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-36-26-7/monitor.h5',
                   'w4': 'results/approx-eval/ps/f32-rds-monitor/_p4000000_b1_s256_t40000_w4_o4_N1_red1_mtw50_seed0_approx2_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-44-50-76/monitor.h5',
                   'w8': 'results/approx-eval/ps/f32-rds-monitor/_p4000000_b1_s256_t40000_w8_o4_N2_red1_mtw50_seed0_approx2_mpiopenmpi4_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-52-03-4/monitor.h5'
                   },
            'title': 'PS F32-RDS 40kT 1x4M P',
            'base': 'results/approx-eval/ps/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
        },

        'ps-40kt-seed': {
            'in': {
                'seed1-': 'results/approx-eval/ps/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-32-35-6/monitor.h5',
                'seed2-': 'results/approx-eval/ps/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-44-08-46/monitor.h5',
                # 'seed3-': 'results/precision-analysis/ps/precision-seed/_p1000000_b1_s256_t40000_w1_o14_N1_red1_mtw50_seed3_approx0_mpimpich3_lbreportonly_lba500_monitor100_tp0_precdouble_/29Apr20.13-33-05-20/monitor.h5',
            },
            'title': 'PS 40kT seeds 1x4M P',

            'base': 'results/approx-eval/ps/exact/_p4000000_b1_s256_t40000_w1_o4_N1_red1_mtw50_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/25Aug20.10-21-36-33/monitor.h5',
        },
        'sps-f32': {
            'in': {'f32': 'results/approx-eval/sps/float32/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/27Aug20.16-13-09-11/monitor.h5',
                   },
            'title': 'SPS F32 40kT 1x4M P',
            'base': 'results/approx-eval/sps/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
        },
        'sps-srp': {
            'in': {'r2': 'results/approx-eval/sps/srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red2_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-54-44-38/monitor.h5',
                   'r3': 'results/approx-eval/sps/srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-20-10-24/monitor.h5',
                   'r4': 'results/approx-eval/sps/srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red4_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-41-26-15/monitor.h5'},
            'title': 'SPS SRP 40kT 1x4M P',
            'base': 'results/approx-eval/sps/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
        },
        'sps-rds': {
            'in': {'w2': 'results/approx-eval/sps/rds-monitor/_p4000000_b1_s1408_t40000_w2_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.16-01-36-53/monitor.h5',
                   'w4': 'results/approx-eval/sps/rds-monitor/_p4000000_b1_s1408_t40000_w4_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.16-30-49-28/monitor.h5',
                   'w8': 'results/approx-eval/sps/rds-monitor/_p4000000_b1_s1408_t40000_w8_o4_N2_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.16-57-27-55/monitor.h5'
                   },
            'title': 'SPS RDS 40kT 1x4M P',
            'base': 'results/approx-eval/sps/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
            # 'base': 'results/approx-eval/sps/exact/_p1000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-52-48-93/monitor.h5',
        },
        'sps-f32-srp': {
            'in': {'r2': 'results/approx-eval/sps/f32-srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red2_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-06-58-33/monitor.h5',
                   'r3': 'results/approx-eval/sps/f32-srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-25-41-5/monitor.h5',
                   'r4': 'results/approx-eval/sps/f32-srp-monitor/_p4000000_b1_s1408_t40000_w1_o4_N1_red4_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-41-14-8/monitor.h5'},
            'title': 'SPS F32-SRP 40kT 1x4M P',
            'base': 'results/approx-eval/sps/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
        },
        'sps-f32-rds': {
            'in': {'w2': 'results/approx-eval/sps/f32-rds-monitor/_p4000000_b1_s1408_t40000_w2_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-54-53-17/monitor.h5',
                   'w4': 'results/approx-eval/sps/f32-rds-monitor/_p4000000_b1_s1408_t40000_w4_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.13-13-23-79/monitor.h5',
                   'w8': 'results/approx-eval/sps/f32-rds-monitor/_p4000000_b1_s1408_t40000_w8_o4_N2_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.13-29-13-91/monitor.h5'
                   },
            'base': 'results/approx-eval/sps/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
            # 'base': 'results/approx-eval/sps/exact/_p1000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-52-48-93/monitor.h5',
        },
        'sps-40kt-seed': {
            'in': {
                'seed1-': 'results/approx-eval/sps/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.18-00-58-23/monitor.h5',
                'seed2-': 'results/approx-eval/sps/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.18-27-07-24/monitor.h5',
                # 'seed3-': 'results/approx-eval/sps/exact/_p1000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed3_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.13-42-23-86/monitor.h5',
            },
            'title': 'SPS 40kT seeds 1x4M P',
            'base': 'results/approx-eval/sps/exact/_p4000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.17-32-07-77/monitor.h5',
            # 'base': 'results/approx-eval/sps/exact/_p1000000_b1_s1408_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-52-48-93/monitor.h5',
        },
        'lhc-f32': {
            'in': {'f32': 'results/approx-eval/lhc/float32/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/27Aug20.16-13-01-28/monitor.h5',
                   },
            'title': 'LHC F32 40kT 1x4M P',
            'base': 'results/approx-eval/lhc/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
        },
        'lhc-f32-srp': {
            'in': {'r2': 'results/approx-eval/lhc/f32-srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red2_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-25-20-38/monitor.h5',
                   'r3': 'results/approx-eval/lhc/f32-srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-33-56-37/monitor.h5',
                   'r4': 'results/approx-eval/lhc/f32-srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red4_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-41-38-83/monitor.h5'},
            'title': 'LHC F32-SRP 40kT 1x4M P',
            'base': 'results/approx-eval/lhc/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
        },
        'lhc-f32-rds': {
            'in': {'w2': 'results/approx-eval/lhc/f32-rds-monitor/_p4000000_b1_s1000_t40000_w2_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-06-43-76/monitor.h5',
                   'w4': 'results/approx-eval/lhc/f32-rds-monitor/_p4000000_b1_s1000_t40000_w4_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-13-30-88/monitor.h5',
                   'w8': 'results/approx-eval/lhc/f32-rds-monitor/_p4000000_b1_s1000_t40000_w8_o4_N2_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precsingle_artdeloff_/28Sep20.12-19-16-89/monitor.h5'
                   },
            'title': 'LHC F32-RDS 40kT 1x4M P',
            'base': 'results/approx-eval/lhc/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
        },
        'lhc-srp': {
            'in': {'r2': 'results/approx-eval/lhc/srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red2_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-44-53-65/monitor.h5',
                   'r3': 'results/approx-eval/lhc/srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red3_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.14-54-47-86/monitor.h5',
                   'r4': 'results/approx-eval/lhc/srp-monitor/_p4000000_b1_s1000_t40000_w1_o4_N1_red4_mtw0_seed0_approx1_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-05-03-50/monitor.h5'},
            'title': 'LHC SRP 40kT 1x4M P',
            'base': 'results/approx-eval/lhc/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
            # 'base': 'results/approx-eval/lhc/exact/_p1000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-53-16-60/monitor.h5',
        },
        'lhc-rds': {
            'in': {'w2': 'results/approx-eval/lhc/rds-monitor/_p4000000_b1_s1000_t40000_w2_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-15-04-0/monitor.h5',
                   'w4': 'results/approx-eval/lhc/rds-monitor/_p4000000_b1_s1000_t40000_w4_o4_N1_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-23-15-25/monitor.h5',
                   'w8': 'results/approx-eval/lhc/rds-monitor/_p4000000_b1_s1000_t40000_w8_o4_N2_red1_mtw0_seed0_approx2_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-30-38-48/monitor.h5'
                   },
            'title': 'LHC RDS 40kT 1x4M P',
            'base': 'results/approx-eval/lhc/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',

            # 'base': 'results/approx-eval/lhc/exact/_p1000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.11-53-16-60/monitor.h5',
        },
        'lhc-40kt-seed': {
            'in': {
                'seed1-': 'results/approx-eval/lhc/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed1_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-55-09-57/monitor.h5',
                'seed2-': 'results/approx-eval/lhc/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed2_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.16-09-29-31/monitor.h5',
                # 'seed3-': 'results/approx-eval/lhc/exact/_p1000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed3_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/10Aug20.13-01-24-25/monitor.h5',
            },
            'title': 'LHC 40kT seeds 1x4M P',
            'base': 'results/approx-eval/lhc/exact/_p4000000_b1_s1000_t40000_w1_o4_N1_red1_mtw0_seed0_approx0_mpimpich3_lbreportonly,,,,_monitor100_tp0_precdouble_artdeloff_/24Aug20.15-39-18-76/monitor.h5',
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

    last_t = args.turns
    outdir = args.outdir
    points = args.points
    # tss = args.ts

    for inputkey in gconfig['inputkeys']:
        print(f'{inputkey} Reading inputs')
        # for plot in gconfig['plots']:
        # for case in gconfig['cases']:
        # inputkey = args.inputkey
        infiles = gconfig['plots'][inputkey]['in']
        basefile = gconfig['plots'][inputkey]['base']
        based = {}
        ind = {}

        # Read basefile
        fullfile = os.path.join(project_dir, basefile)
        h5file = h5py.File(fullfile, 'r')
        for key in h5file[gconfig['group']]:
            val = h5file[gconfig['group']][key][()]
            if key not in based:
                based[key] = val.reshape(len(val))
        h5file.close()
        turns = based[gconfig['x_name']]
        del based[gconfig['x_name']]

        # Read infile
        for keyf, infile in infiles.items():
            fullfile = os.path.join(project_dir, infile)
            h5file = h5py.File(fullfile, 'r')
            if keyf not in ind:
                ind[keyf] = {}
            for key in h5file[gconfig['group']]:
                val = h5file[gconfig['group']][key][()]
                if key not in ind:
                    ind[keyf][key] = val.reshape(len(val))
            h5file.close()
            assert np.array_equal(turns, ind[keyf][gconfig['x_name']])
            del ind[keyf][gconfig['x_name']]

        points = min(len(turns), points) if points > 0 else len(turns)
        intv = int(np.ceil(len(turns)/points))

        print(f'{inputkey} Plotting')

        fig, ax = plt.subplots(ncols=1, nrows=1,
                               sharex=True, sharey=True,
                               figsize=gconfig['figsize'])
        for keyf in ind.keys():
            for key in (set(based.keys()) & set(ind[keyf].keys())):
                if key not in gconfig['y_names']:
                    continue
                basevals = based[key]
                indvals = ind[keyf][key]
                assert len(basevals) == len(
                    indvals) and len(turns) == len(basevals)

                error = np.abs(1 - indvals / basevals)
                plt.plot(turns[::intv], error[::intv],
                         label='{}{}'.format(keyf, gconfig['labels'][key]),
                         )
                # marker=gconfig['markers'][idx],
                # color=gconfig['colors'][idx],
                # yerr=yerr,
                # capsize=2)
        plt.yscale('log')
        # ax.set_yscale('log')
        # plt.rcParams['axes.formatter.min_exponent'] = 1
        plt.grid(True, which='both', axis='y', alpha=0.5)
        plt.grid(False, which='major', axis='x')
        # plt.title(gconfig['plots'][inputkey]['title'], **gconfig['title'])
        plt.xlabel(**gconfig['xlabel'])
        plt.ylabel(**gconfig['ylabel'])
        plt.ylim(gconfig['ylim'])
        # plt.gca().ticklabel_format(axis='y', style='sci')
        # plt.xlim(gconfig['xlim'])
        # plt.xticks(x//20, np.array(x, int)//20, **gconfig['ticks'])
        ax.tick_params(**gconfig['tick_params'])
        ax.legend(**gconfig['legend'])

        plt.xticks(**gconfig['ticks'])
        yticks = [10**i for i in range(int(np.log10(gconfig['ylim'][0])),
                                       int(np.log10(gconfig['ylim'][1]))+1)]
        plt.yticks(yticks, ['{:1.0e}'.format(yi)
                            for yi in yticks], **gconfig['ticks'])

        plt.tight_layout()
        plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir, this_filename[: -3], inputkey)
            print('[{}] {}: {}'.format(
                this_filename[: -3], 'Saving figure', file))

            save_and_crop(fig, file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()

    # for dirpath, dirnames, filenames in os.walk(indir):
    #     if 'monitor.h5' not in filenames:
    #         continue

    #     particles = dirpath.split('_p')[1].split('_')[0]
    #     bunches = dirpath.split('_b')[1].split('_')[0]
    #     seed = dirpath.split('_seed')[1].split('_')[0]
    #     monitor_intv = dirpath.split('_m')[1].split('_')[0]
    #     red = dirpath.split('_r')[1].split('_')[0]
    #     workers = dirpath.split('_w')[1].split('_')[0]

    #     fullfile = dirpath + '/monitor.h5'
    #     inh5file = h5py.File(fullfile, 'r')
    #     x = inh5file['default']['turns'].value
    #     # intv = int(np.ceil(len(x)/points))
    #     for key in inh5file['default'].keys():
    #         if (key in not_plot):
    #             continue
    #         val = inh5file['default'][key].value
    #         if (key == 'profile'):
    #             val = val[-1]
    #         val = val.reshape(len(val))
    #         if (key not in plot_dir):
    #             plot_dir[key] = {}
    #         if (workers not in plot_dir[key]):
    #             plot_dir[key][workers] = {'num': 0}
    #         if ('sum' not in plot_dir[key][workers]):
    #             plot_dir[key][workers]['sum'] = np.zeros_like(val)
    #             plot_dir[key][workers]['min'] = val
    #             plot_dir[key][workers]['max'] = val
    #         plot_dir[key][workers]['num'] += 1
    #         plot_dir[key][workers]['sum'] += val
    #         plot_dir[key][workers]['min'] = np.minimum(
    #             plot_dir[key][workers]['min'], val)
    #         plot_dir[key][workers]['max'] = np.maximum(
    #             plot_dir[key][workers]['max'], val)
    #         plot_dir[key][workers]['turns'] = x
    #     inh5file.close()

    # # continue here, I need to iterate over the errors, create a figure for each
    # # iterate over the reduce values, add an error plot line for each acording to the intv etc

    # for ts in tss:
    #     # filename = outfile + '/ts' + str(ts) + '.h5'
    #     if not os.path.exists(os.path.dirname(outdir + '/ts' + ts+'/')):
    #         os.makedirs(os.path.dirname(outdir + '/ts' + ts+'/'))
    #     lines = 0
    #     for error in plot_dir.keys():
    #         fig = plt.figure(figsize=(4, 4))
    #         outfiles = [
    #             # '{}/ts{}/{}.pdf'.format(outdir, ts, error),
    #             '{}/ts{}/{}.jpeg'.format(outdir, ts, error)]

    #         plt.grid()
    #         if args.get('ymin', None):
    #             plt.ylim(ymin=args.ymin)
    #         if args.get('ymax', None):
    #             plt.ylim(ymax=args.ymax)

    #         plt.title('Ts: {}, Variable: {}'.format(ts, error))
    #         plt.xlabel('#Turn')
    #         plt.ylabel('Raw value')

    #         for workers, data in plot_dir[error].items():
    #             x = data['turns']
    #             y = data['sum'] / data['num']
    #             ymin = data['min']
    #             ymax = data['max']
    #             intv = int(np.ceil(len(x)/points))
    #             label = 'r{}'.format(workers)
    #             marker = markers.get('r{}'.format(workers), None)
    #             if (error == 'profile'):
    #                 # y = y[-1]
    #                 nonzero = np.flatnonzero(ymax)
    #                 y = y[nonzero[0]:nonzero[-1]]
    #                 ymin = ymin[nonzero[0]:nonzero[-1]]
    #                 ymax = ymax[nonzero[0]:nonzero[-1]]
    #                 x = np.arange(len(y))
    #                 plt.xlabel('#Bin')
    #             else:
    #                 y = running_mean(y, int(ts))
    #                 ymin = running_mean(ymin, int(ts))
    #                 ymax = running_mean(ymax, int(ts))
    #                 x = x[:len(y):intv]
    #                 y = y[::intv]
    #                 ymin = ymin[::intv]
    #                 ymax = ymax[::intv]
    #             if (workers == '1'):
    #                 plt.fill_between(
    #                     x, ymin, ymax, facecolor='0.6', interpolate=True)
    #                 plt.plot(x, ymax, color='black', linewidth=1)
    #                 plt.plot(x, ymin, color='black',  linewidth=1, label=label)
    #             else:
    #                 plt.errorbar(x, y,
    #                              yerr=[y-ymin, ymax-y],
    #                              label=label, linestyle='--',
    #                              marker=marker, markersize=0,
    #                              # color=next(colors),
    #                              # alpha=0.5,
    #                              capsize=1, elinewidth=1)
    #             lines += 1
    #         plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #         plt.legend(loc='upper left', fancybox=True, fontsize=9,
    #                        ncol=(lines+2)//3, columnspacing=1,
    #                        labelspacing=0.1, borderpad=0.2, framealpha=0.5,
    #                        handletextpad=0.2, handlelength=1.5, borderaxespad=0)

    #         plt.tight_layout()
    #         for outfile in outfiles:
    #             save_and_crop(fig, outfile, dpi=900, bbox_inches='tight')
    #             # fig.savefig(outfile, dpi=900, bbox_inches='tight')
    #         if args.show is True:
    #             plt.show()
    #         plt.close()
