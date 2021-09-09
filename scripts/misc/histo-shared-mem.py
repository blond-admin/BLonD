import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse

# python scripts/misc/histo-shared-mem.py -i=results/misc/histo_sm/histo_sm_onoff.csv -o results/misc/plots/ -s

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the intermediate effect analysis.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-i', '--infile', type=str, default=os.path.join(project_dir, 'results/misc/gpu-cache-on-off.csv'),
                    help='The directory with the results.')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='The directory to store the plots.'
                    'Default: In a plots directory inside the input results directory.')


parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')


args = parser.parse_args()

if args.outdir is None:
    images_dir = 'temp'
else:
    images_dir = args.outdir

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

gconfig = {
    'label': {
        'on': 'With Shared Memory',
        'off': 'Without Shared Memory',
    },
    'colors': {
        'on': '0.2',
        'off': '0.8',
    },
    'hatches': {
        'on': '',
        'off': '',
    },

    'x_name': 'size(1e6)',
    'y_name': 'time',
    'xlabel': {
        'xlabel': r'Number of Macro-Partciles (x$10^6$)'
    },
    'ylabel': 'Norm. Runtime',
    'title': {
        # 's': '',
        'fontsize': 10,
        'y': .96,
        'x': 0.1,
        'fontweight': 'bold',
    },
    'figsize': [5, 2.],
    'annotate': {
        'fontsize': 10,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'xticks': {'fontsize': 10, 'rotation': '0', 'fontweight': 'bold'},
    'ticks': {'fontsize': 10, 'rotation': '0'},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 9, 'handlelength': 1.6, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.2,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.5,
        # 'bbox_to_anchor': (-0.01, 1.)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 0, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    'ylim': [0.0, 1.2],
    'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1],
    'outfiles': [
        '{}/{}.png',
        '{}/{}.pdf'
    ],
    'lines': {
        'sharedmem': ['on', 'off'],
    }

}

# Force sans-serif math mode (for axes labels)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}'
plt.rcParams['font.family'] = gconfig['fontname']


if __name__ == '__main__':
    # read the results
    plots_dir = {}
    data = np.genfromtxt(args.infile, delimiter='\t', dtype=str)
    header, data = list(data[0]), data[1:]
    temp = get_plots(header, data, gconfig['lines'],
                     exclude=gconfig.get('exclude', []),
                     prefix=True)
    for k in temp.keys():
        sm = k.split('sharedmem')[1].split('_')[0]
        plots_dir[sm] = temp[k]

    print(plots_dir)

    fig, ax = plt.subplots(ncols=1, nrows=1,
                           sharex=True, sharey=True,
                           figsize=gconfig['figsize'])

    plt.sca(ax)
    plt.ylabel(gconfig['ylabel'], labelpad=3,
               fontweight='bold',
               fontsize=gconfig['fontsize'])

    pos = 0
    labels = set()
    xticks = []
    xtickspos = []
    # width = 1.
    width = .8 / (len(plots_dir))
    for sm, vals in plots_dir.items():
        sizes = get_values(vals, header, gconfig['x_name'])
        times = get_values(vals, header, gconfig['y_name'])
        reftimes = get_values(plots_dir['off'], header, gconfig['y_name'])
        normtime = times/reftimes
        label = sm
        if label in labels:
            legend = None
        else:
            legend = gconfig['label'][label]
            labels.add(label)
        x = pos + np.arange(len(sizes))
        plt.bar(x, normtime,
                width=0.9*width,
                edgecolor='0.', label=legend,
                hatch=gconfig['hatches'][label],
                color=gconfig['colors'][label])
        if sm == 'on':
            for xi, nt in zip(x, normtime):
                ax.annotate('{:.2f}'.format(nt),
                            xy=(xi, nt),
                            rotation='0',
                             **gconfig['annotate'])
        pos += width

    xtickspos += list(width/2+np.arange(len(sizes)))
    xticks += list(np.array(sizes, int))
        # ax.annotate(sm, xy=(pos-2.5*width + len(sizes)/2, 1.),
        #             rotation='0', **gconfig['annotate'])
        # pos += len(sizes) - width

    plt.grid(True, which='major', alpha=0.5)
    plt.grid(False, which='major', axis='x')
    plt.gca().set_axisbelow(True)


    plt.legend(**gconfig['legend'])

    plt.ylim(gconfig['ylim'])
    plt.yticks(gconfig['yticks'], **gconfig['ticks'])
    plt.xticks(xtickspos, xticks, **gconfig['ticks'])
    plt.xlabel(**gconfig['xlabel'])
    # plt.xlim(0, pos-width)

    ax.tick_params(**gconfig['tick_params'])
    plt.tight_layout()
    # plt.subplots_adjust(**gconfig['subplots_adjust'])

    for file in gconfig['outfiles']:
        file = file.format(
            images_dir, this_filename[:-3])
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        save_and_crop(fig, file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()

