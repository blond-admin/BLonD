import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker
import sys
from plot.plotting_utilities import *
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the MPI libraries benchmark.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-i', '--inputdir', type=str, default=os.path.join(project_dir, 'results'),
                    help='The directory with the results.')

parser.add_argument('-c', '--cases', type=str, default='lhc,sps,ps',
                    help='A comma separated list of the testcases to run. Default: lhc,sps,ps')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')

args = parser.parse_args()
args.cases = args.cases.split(',')

res_dir = args.inputdir
images_dir = os.path.join(res_dir, 'plots')

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

gconfig = {
    'approx': {
        '0': 'exact',
        '1': 'SMD',
        '2': 'RDS',
    },
    'hatches': ['', '', ''],
    'colors': ['0.1', '0.5', '0.8'],
    'x_name': 'n',
    'x_to_keep': [8],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    'xlabel': '',
    'ylabel': 'Norm. Runtime',
    'title': {
                's': '',
                'fontsize': 10,
                'y': 0.74,
                # 'x': 0.55,
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
        'loc': 'upper left', 'ncol': 3, 'handlelength': 1.5, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
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
    'ylim': [.9, 1.4],
    'yticks': [.9, 1., 1.1, 1.2, 1.3, 1.4],
    'outfiles': ['{}/{}-{}.png'],
    'files': [
        '{}/{}/approx0-impl/comm-comp-report.csv',
    ],
    'lines': {
        'mpi': ['mvapich2', 'mpich3', 'openmpi3'],
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
    plt.ylabel(gconfig['ylabel'], labelpad=3, color='xkcd:black',
               fontweight='bold',
               fontsize=gconfig['fontsize'])

    pos = 0
    step = 1.
    colors1 = ['0.2', '0.5', '0.8']
    labels = set()
    avg = {}

    for col, case in enumerate(args.cases):
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Reading data'))

        # ax2 = ax.twinx()
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
        refvals = plots_dir[keyref]

        x = get_values(refvals, header, gconfig['x_name'])
        omp = get_values(refvals, header, gconfig['omp_name'])
        y = get_values(refvals, header, gconfig['y_name'])
        parts = get_values(refvals, header, 'ppb')
        bunches = get_values(refvals, header, 'b')
        turns = get_values(refvals, header, 't')
        yref = parts * bunches * turns / y

        width = 0.9 * step / (len(plots_dir.keys()))
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))

        for idx, k in enumerate(plots_dir.keys()):
            values = plots_dir[k]
            mpiv = k.split('_mpi')[1].split('_')[0]
            experiment = k.split('_')[-1]
            label = '{}'.format(mpiv)
            if label in labels:
                label = None
            else:
                labels.add(label)

            x = get_values(values, header, gconfig['x_name'])
            omp = get_values(values, header, gconfig['omp_name'])
            y = get_values(values, header, gconfig['y_name'])
            parts = get_values(values, header, 'ppb')
            bunches = get_values(values, header, 'b')
            turns = get_values(values, header, 't')

            # This is the throughput
            y = parts * bunches * turns / y

            normtime = yref / y
            if len(x) > 1:
                x_new = []
                sp_new = []

                for i, xi in enumerate(gconfig['x_to_keep']):
                    if xi in x:
                        x_new.append(xi)
                        sp_new.append(normtime[list(x).index(xi)])
                    # else:
                        # sp_new.append(0)
                x = np.array(x_new)
                normtime = np.array(sp_new)
            x = x * omp[0]

            if mpiv not in avg:
                avg[mpiv] = []
            avg[mpiv].append(normtime)

            # efficiency = 100 * speedup / x
            plt.bar(pos + width * idx, normtime, width=0.9*width,
                    edgecolor='0.', label=label, hatch=gconfig['hatches'][idx],
                    color=gconfig['colors'][idx])
        pos += step
    # I plot the averages here

    for idx, key in enumerate(avg.keys()):
        val = np.mean(avg[key])
        plt.bar(pos + idx*width, val, width=0.9*width,
                edgecolor='0.', label=None, hatch=gconfig['hatches'][idx],
                color=gconfig['colors'][idx])
        ax.annotate('{:.2f}'.format(val), xy=(pos + idx*width, val),
                    **gconfig['annotate'])
    pos += step

    plt.xticks(np.arange(pos) + width, [c.upper()
                                        for c in args.cases] + ['AVG'])

    plt.ylim(gconfig['ylim'])
    plt.legend(**gconfig['legend'])
    ax.tick_params(**gconfig['tick_params'])
    plt.xticks(**gconfig['ticks'], fontweight='bold')
    plt.yticks(gconfig['yticks'], **gconfig['ticks'])

    plt.tight_layout()
    plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(images_dir, this_filename[:-3], '-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        fig.savefig(file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
