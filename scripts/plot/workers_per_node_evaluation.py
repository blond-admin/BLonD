import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the MPI workers per node evaluation.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-i', '--inputdir', type=str, default=os.path.join(project_dir, 'results/local'),
                    help='The directory with the results.')

parser.add_argument('-c', '--cases', type=str, default='lhc,sps,ps',
                    help='A comma separated list of the testcases to run. Default: lhc,sps,ps')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')
parser.add_argument('-e', '--errorbars', action='store_true',
                    help='Add errorbars.')


args = parser.parse_args()
args.cases = args.cases.split(',')

res_dir = args.inputdir
images_dir = os.path.join(res_dir, 'plots')

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

gconfig = {
    'hatches': ['', '', 'xx', '', 'xx'],
    'colors': ['0.3', '0.6', '0.6', '0.9', '0.9'],
    'x_name': 'omp',
    # 'x_to_keep': [2, 5, 10, 20],
    # 'x_to_keep': [8, 16],
    'omp_name': 'n',
    'y_name': 'avg_time(sec)',
    # 'y_err_name': 'std',
    'xlabel': '',
    'ylabel': 'Norm. Runtime',
    'title': {
        's': '',
        'fontsize': 10,
        'y': 0.74,
        # 'x': 0.55,
        'fontweight': 'bold',
    },
    'figsize': [5, 2.1],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'title_annotate': {
        'label': 'Workers-Per-Node:',
    },
    'ticks': {'fontsize': 10},
    'xticks': {'fontsize': 10, 'rotation': '0', 'fontweight': 'bold'},
    'fontsize': 10,
    'legend': {
        'loc': 'upper right', 'ncol': 5, 'handlelength': 1.5, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        # 'title': 'Worker-Per-Node',
        # 'bbox_to_anchor': (0, 1.25)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 0, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    'ylim': [.5, 1.1],
    'yticks': [.5, .6, .7, .8, .9, 1.],
    'outfiles': ['{}/{}-{}.png'],
    'files': [
        '{}/{}/approx0-workers/comm-comp-report.csv',
    ],
    'lines': {
        'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
        'type': ['total'],
    }

}

plt.rcParams['font.family'] = gconfig['fontname']
# plt.rcParams['text.usetex'] = True


if __name__ == '__main__':
    fig, ax = plt.subplots(ncols=1, nrows=1,
                           sharex=True, sharey=True,
                           figsize=gconfig['figsize'])
    plt.sca(ax)
    plt.plot([], [], ls=' ', **gconfig['title_annotate'])
    plt.xlabel(gconfig['xlabel'], labelpad=3,
               fontsize=gconfig['fontsize'])
    plt.ylabel(gconfig['ylabel'], labelpad=3, color='xkcd:black',
               fontweight='bold',
               fontsize=gconfig['fontsize'])
    # ax_arr = np.atleast_1d(ax_arr)
    pos = 0
    step = 1.
    labels = set()
    avg = []
    xticks = []
    xtickspos = []
    for col, case in enumerate(args.cases):
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Reading data'))

        plots_dir = {}
        for file in gconfig['files']:
            file = file.format(res_dir, case)
            # print(file)
            data = np.genfromtxt(file, delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = get_plots(header, data, gconfig['lines'],
                             exclude=gconfig.get('exclude', []),
                             prefix=True)
            for key in temp.keys():
                plots_dir['_{}_'.format(key)] = temp[key].copy()

        # First the reference value
        keyref = ''
        for k in plots_dir.keys():
            if 'mvapich2' in k:
                keyref = k
                break
        if keyref == '':
            print('ERROR: mvapich2 not found')
            exit(-1)

        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))

        for idx, k in enumerate(plots_dir.keys()):
            values = plots_dir[k]

            x = get_values(values, header, gconfig['x_name'])
            x = np.array(x, int)
            sortidx = np.argsort(x)
            x = x[sortidx]
            omp = get_values(values, header, gconfig['omp_name'])[sortidx]
            y = get_values(values, header, gconfig['y_name'])[sortidx]
            parts = get_values(values, header, 'ppb')[sortidx]
            bunches = get_values(values, header, 'b')[sortidx]
            turns = get_values(values, header, 't')[sortidx]


            # This is the throughput
            y = parts * bunches * turns / y
            speedup = y
            # x_new = []
            # sp_new = []
            # omp_new = []
            # for i, xi in enumerate(gconfig['x_to_keep']):
            #     x_new.append(xi)
            #     if xi in x:
            #         sp_new.append(speedup[list(x).index(xi)])
            #         omp_new.append(omp[list(x).index(xi)])
            #     else:
            #         sp_new.append(0)
            #         omp_new.append(0)
            # x = np.array(x_new)
            # omp = np.array(omp_new)
            # speedup = np.array(sp_new)

            # efficiency = 100 * speedup / (x * omp / ompref)
            # x = x * omp
            speedup = speedup[0] / speedup

            width = .9 * step / (len(x))
            avg.append(speedup)
            # efficiency = 100 * speedup / x
            for ii, sp in enumerate(speedup):
                plt.bar(pos + width*ii, sp, width=0.9*width,
                        edgecolor='0.', label=None, hatch=gconfig['hatches'][ii],
                        color=gconfig['colors'][ii])
        pos += step
        # I plot the averages here

    vals = np.mean(avg, axis=0)
    for idx, val in enumerate(vals):
        plt.bar(pos + idx*width, val, width=0.9*width,
                edgecolor='0.', label=str(int(20//x[idx])), hatch=gconfig['hatches'][idx],
                color=gconfig['colors'][idx])
        text = '{:.2f}'.format(val)
        if idx == 0:
            text = ''
        else:
            text = text[1:]
        ax.annotate(text, xy=(pos + idx*width, val),
                    **gconfig['annotate'])

    plt.ylim(gconfig['ylim'])
    pos += step
    plt.xlim(0-step/6, pos-step/7)
    plt.xticks(np.arange(pos) + step/2,
               [c.upper() for c in args.cases] + ['AVG'], **gconfig['xticks'])

    plt.legend(**gconfig['legend'])
    plt.yticks(gconfig['yticks'], **gconfig['ticks'])
    ax.tick_params(**gconfig['tick_params'])

    plt.tight_layout()
    plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(
            images_dir, this_filename[:-3], '-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        fig.savefig(file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
