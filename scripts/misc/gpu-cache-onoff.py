import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse

# python scripts/misc/gpu-cache-onoff.py -i=results/misc/gpu-cache-on-off.csv -o results/misc/plots -s

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
        'on': 'With Memory Pool',
        'off': 'Without Memory Pool',
    },
    'colors': {
        'on': 'tab:blue',
        'off': 'tab:orange',
    },
    'hatches': {
        'on': '',
        'off': '',
    },

    'x_name': 'size(M)',
    'y_name': 'time',
    'xlabel': {
        'xlabel': r'Number of Macro-Particles (x$10^6$)'
    },
    'ylabel': 'Norm. Runtime',
    'title': {
        # 's': '',
        'fontsize': 10,
        'y': .98,
        'x': 0.1,
        'fontweight': 'bold',
    },
    'figsize': [5, 2.8],
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
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.5,
        'bbox_to_anchor': (-0.01, 1.12)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 0, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    'ylim': [0.4, 1.15],
    'yticks': [0.4, 0.6, 0.8, 1],
    'outfiles': [
        '{}/{}.png',
        '{}/{}.pdf'
    ],
    'lines': {
        'testcase': ['LHC', 'SPS', 'PS'],
        'cache': ['on', 'off'],
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
    # file = file.format(res_dir, case)
    # print(file)
    data = np.genfromtxt(args.infile, delimiter='\t', dtype=str)
    header, data = list(data[0]), data[1:]
    temp = get_plots(header, data, gconfig['lines'],
                     exclude=gconfig.get('exclude', []),
                     prefix=True)
    for k in temp.keys():
        tc = k.split('testcase')[1].split('_')[0]
        cache = k.split('cache')[1].split('_')[0]
        if tc not in plots_dir:
            plots_dir[tc] = {}
        plots_dir[tc][cache] = temp[k]
    # plots_dir.update(temp)
    print(plots_dir)

    fig, ax = plt.subplots(ncols=1, nrows=1,
                           sharex=True, sharey=True,
                           figsize=gconfig['figsize'])

    plt.sca(ax)
    plt.ylabel(gconfig['ylabel'], labelpad=3,
               fontweight='bold',
               fontsize=gconfig['fontsize'])

    pos = 0
    step = 1.
    labels = set()
    avg = {}
    xticks = []
    xtickspos = []

    for tc in plots_dir.keys():
        width = .9 / (len(plots_dir[tc]))
        for cache, vals in plots_dir[tc].items():
            sizes = get_values(vals, header, gconfig['x_name'])
            times = get_values(vals, header, gconfig['y_name'])
            reftimes = get_values(
                plots_dir[tc]['off'], header, gconfig['y_name'])
            normtime = times/reftimes
            label = cache
            if label in labels:
                legend = None
            else:
                legend = gconfig['label'][label]
                labels.add(label)
            x = pos + np.arange(len(sizes))
            plt.bar(x, normtime,
                    width=0.85*width,
                    edgecolor='0.', label=legend,
                    hatch=gconfig['hatches'][label],
                    color=gconfig['colors'][label])
            if label not in avg:
                avg[label] = []

            avg[label].append(normtime)
            # if cache == 'on':
            #     for xi, nt in zip(x, normtime):
            #         ax.annotate('{:.2f}'.format(nt),
            #                     xy=(xi, nt),
            #                     rotation='90',
            #                      **gconfig['annotate'])

            pos += width
        xtickspos += list(pos - 0.75 + np.arange(len(sizes)))
        xticks += list(np.array(sizes, int))
        ax.annotate(tc, xy=(pos-2.5*width + len(sizes)/2, 1.05),
                    rotation='0', **gconfig['annotate'])
        pos += len(sizes) - 1.5 * width
        plt.axvline(x=pos-0.75*width, color='black', ls='--')

    # xticks.append('AVG')
    for idx, key in enumerate(avg.keys()):
        vals = avg[key]
        val = np.mean(vals, axis=0)
        x = pos + np.arange(len(sizes))
        plt.bar(x, val, width=.85 * width,
                edgecolor='0.', label=None,
                hatch=gconfig['hatches'][key],
                color=gconfig['colors'][key])
        if key == 'on':
            for xi, nt in zip(x, val):
                ax.annotate('{:.2f}'.format(nt),
                            xy=(xi, nt),
                            rotation='90',
                             **gconfig['annotate'])
        pos += width

    xtickspos += list(pos - 0.75 + np.arange(len(sizes)))
    ax.annotate('AVG', xy=(pos-2.5*width + len(sizes)/2, 1.05),
                rotation='0', **gconfig['annotate'])

    plt.grid(True, which='major', alpha=0.5)
    plt.grid(False, which='major', axis='x')
    plt.gca().set_axisbelow(True)

    # plt.title('{}'.format(case.upper()), **gconfig['title'])

    plt.legend(**gconfig['legend'])

    plt.ylim(gconfig['ylim'])
    plt.yticks(gconfig['yticks'], **gconfig['ticks'])
    # plt.xticks(xtickspos, xticks, **gconfig['ticks'])
    plt.xticks(xtickspos, [12, 24, 36]*4, **gconfig['ticks'])
    plt.xlabel(**gconfig['xlabel'])
    plt.xlim(0-width, xtickspos[-1]+1.5*width)

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

