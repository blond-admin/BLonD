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
    description='Plot multiple beam profiles in a single plot.')

parser.add_argument('-i', '--input', type=str,
                    help='Input file.')

# parser.add_argument('-c', '--cases', nargs='+', choices=['lhc', 'sps', 'ps'],
# default=['lhc', 'sps', 'ps'],
# help='Which testcases to plot.')

# parser.add_argument('-t', '--techniques', nargs='+',
# choices=['base', 'rds', 'srp', 'f32', 'f32-srp', 'f32-rds'],
# default=['base', 'f32', 'rds', 'f32-rds', 'srp', 'f32-srp'],
# help='which techniqeus to plot')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='Directory to store the plots.')

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
                    help='Num of profiles to plot. Default: all')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plot or save only. Default: save only')


gconfig = {
    'hatches': ['', '', 'xx'],
    'markers': ['x', 'o', '^'],
    'boxcolor': 'xkcd:light blue',
    'linecolor': 'xkcd:blue',
    'colors': ['xkcd:orange', 'xkcd:green', 'xkcd:blue'],
    # 'colors': ['.7', '0.4', '0.1'],
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
    'xlabel': {'xlabel': r'Time', 'labelpad': 2, 'fontsize': 10},
    'ylabel': {'ylabel': r'Beam Profile', 'labelpad': 2, 'fontsize': 10},
    'title': {
        # 's': '{}'.format(case.upper()),
        'fontsize': 10,
        # 'y': .95,
        # 'x': 0.45,
        'fontweight': 'bold',
    },
    'figsize': [5, 2.],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 10},
    'fontsize': 10,
    'legend': {
        'loc': 'best', 'ncol': 1, 'handlelength': 1.5, 'fancybox': False,
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
    'fontname': 'DejaVu Sans Mono',
    'ylim': [700, 3000],
    'xlim': [38, 84],
    'outfiles': ['{}/{}-{}.png',
                 '{}/{}-{}.pdf'
                 ],
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
    assert args.input, 'You must provide the indir argument'

    res_dir = args.outdir
    images_dir = os.path.join(res_dir)

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    fullfile = args.input

    h5file = h5py.File(fullfile, 'r')
    tempd = {}
    for key in h5file[gconfig['group']]:
        val = h5file[gconfig['group']][key][()]
        if key not in tempd:
            # tempd[key] = val.reshape(len(val))
            tempd[key] = val
        tempd[key] = tempd[key][first_t:]
        if last_t:
            tempd[key] = tempd[key][:last_t]
    h5file.close()
    based = tempd.copy()
    turns = based[gconfig['x_name']]

    points = min(len(turns), points) if points > 0 else len(turns)
    intv = int(np.ceil(len(turns)/points))

    fig, ax = plt.subplots(ncols=1, nrows=1,
                           sharex=True, sharey=True,
                           figsize=gconfig['figsize'])
    offset = 0
    for turn, profile in zip(turns[::intv], tempd['profile'][::intv]):
        # plt.hist(profile, bins=len(profile), alpha=0.5)
        # plt.bar(np.arange(len(profile)), profile,
                 # color=gconfig['colors'][turn % len(gconfig['colors'])],
                 # color=gconfig['colors'][turn % len(gconfig['colors'])],
                 # facecolor='0.5',
                 # edgecolor=gconfig['colors'][turn % len(gconfig['colors'])],
                 # width=0.7,
                 # alpha=0.9,
                 # label='turn-{}'.format(turn))
        # plt.plot(np.arange(len(profile)), profile,
        #          color=gconfig['colors'][turn % len(gconfig['colors'])],
        #          # marker='_', 
        #          ls='-',
        #          label='turn-{}'.format(turn))
        plt.bar(np.arange(len(profile)) + offset, profile,
                 color=gconfig['colors'][turn % len(gconfig['colors'])],
                 # color=gconfig['colors'][turn % len(gconfig['colors'])],
                 # facecolor='0.5',
                 # edgecolor=gconfig['colors'][turn % len(gconfig['colors'])],
                 width=0.25,
                 # alpha=0.9,
                 label='turn-{}'.format(turn))
        offset += 0.25

        # plt.fill_between(np.arange(len(profile)), profile, 0, 
        #     # color=gconfig['colors'][turn % len(gconfig['colors'])],
        #     facecolor='1',
        #     alpha=0.5, hatch='x',
        #     edgecolor=gconfig['colors'][turn % len(gconfig['colors'])])

    # exit()

    plt.grid(False, which='both', axis='both', alpha=0.5)
    # plt.grid(False, which='major', axis='x')
    # plt.title(case.upper(), **gconfig['title'])
    plt.xlabel(**gconfig['xlabel'])
    plt.ylabel(**gconfig['ylabel'])
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlim(gconfig['xlim'])
    plt.ylim(gconfig['ylim'])
    # plt.xticks(xtickspos, xticks, **gconfig['ticks'])
    # yticks = ['{:1.0e}'.format(10**i) for i in range(gconfig['ylim'][0],
                                                     # gconfig['ylim'][1]+1)]
    # plt.yticks(np.arange(gconfig['ylim'][0],
                         # gconfig['ylim'][1]+1), yticks, **gconfig['ticks'])
    # print('{} Median: {:e}'.format(inkey, np.median(errordist)))
    # for patch in bplot['boxes']:
    #     patch.set_facecolor(gconfig['boxcolor'])
    #     patch.set_edgecolor(gconfig['linecolor'])
    # for patch in bplot['medians']:
    #     patch.set_color(gconfig['linecolor'])
    # for patch in bplot['caps']:
    #     patch.set_color(gconfig['linecolor'])
    # for patch in bplot['whiskers']:
    #     patch.set_color(gconfig['linecolor'])

    # xtickspos.append(pos)
    # xticks.append(tech)
    # pos += step
    # plt.yscale('log')
    # ax.tick_params(**gconfig['tick_params'])
    plt.legend(**gconfig['legend'])
    plt.tight_layout()
    # plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(images_dir, this_filename[: -3], args.input.split('/')[-1].replace('.h5', ''))
        print('[{}] {}: {}'.format(
            this_filename[: -3], 'Saving figure', file))

        save_and_crop(fig, file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
