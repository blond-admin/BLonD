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

parser = argparse.ArgumentParser(description='Generate the figure of the load imbalance spread.',
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
        '1': 'SRP',
        '2': 'RDS',
    },
    'hatches': ['', '', '//'],
    'markers': ['x', 'o', '^'],
    'colors': ['0.25', '0.55', '0.9'],
    'ecolor': 'xkcd:red',
    'capsize': 4,
    'x_name': 'n',
    # 'x_to_keep': [4, 8, 16, 32],
    'omp_name': 'omp',
    'y_name': 'total_time(sec)',
    'percent_name': 'global_percentage',
    'xlabel': 'Nodes (x20 Cores)',
    'ylabel': r'Time Spread(%)',
    'title': {
                # 's': '{}'.format(case.upper()),
                'fontsize': 10,
                'y': .85,
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
        'loc': 'upper left', 'ncol': 3, 'handlelength': 2, 'fancybox': False,
        'framealpha': .0, 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        # 'bbox_to_anchor': (0., 0.85)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params_left': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'tick_params_center_right': {
        'pad': 1, 'top': 0, 'bottom': 1, 'left': 0,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',

    # 'ylim': [0, 28],
    'ylim': [0, 40],
    'xlim': [1.6, 36],
    'yticks': [0, 8, 16, 24, 32, 40],
    'outfiles': ['{}/{}-{}.png'],
    'errorfile': 'delta-std-report.csv',
    'deltafile': 'delta-report.csv',
    'datafile': 'avg-report.csv',
    'files': [
        '{}/{}/approx0-spread/{}',
    ],
    'lines': {
        # 'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
        # 'lb': ['interval', 'reportonly'],
        # 'approx': ['0', '1', '2'],
        # 'lba': ['500'],
        # 'b': ['6', '12', '24', '96', '192',
        #       '48', '21', '9', '18', '36',
        #       '72', '144', '288'],
        # 't': ['5000'],
        'function': ['serial:sync'],
    }

}

plt.rcParams['font.family'] = gconfig['fontname']
# plt.rcParams['text.usetex'] = True

if __name__ == '__main__':
    fig, ax = plt.subplots(ncols=1, nrows=1,
                           sharex=True, sharey=True,
                           figsize=gconfig['figsize'])
    plt.sca(ax)
    plt.grid(True, which='both', axis='y', alpha=0.5)
    # plt.grid(True, which='minor', alpha=0.5, zorder=1)
    plt.grid(False, which='major', axis='x')
    # plt.title('{}'.format(case.upper()), **gconfig['title'])
    plt.xlabel(gconfig['xlabel'], labelpad=3,
               fontweight='bold',
               fontsize=gconfig['fontsize'])
    plt.ylabel(gconfig['ylabel'], labelpad=3,
               fontweight='bold',
               fontsize=gconfig['fontsize'])
    labels = set()
    pos = 0
    step = 1
    width = 0.9 * step / len(args.cases)
    for col, case in enumerate(args.cases):
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Reading data'))

        plots_dir = {}
        errors_dir = {}
        deltas_dir = {}
        for file in gconfig['files']:
            # print(file)
            # Here I get the overall time
            data = np.genfromtxt(file.format(res_dir, case, gconfig['datafile']),
                                 delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = get_plots(header, data, gconfig['lines'],
                             exclude=gconfig.get('exclude', []),
                             prefix=True)
            for key in temp.keys():
                plots_dir['_{}'.format(key)] = temp[key].copy()

            # Here I extract the delta
            data = np.genfromtxt(file.format(res_dir, case, gconfig['deltafile']),
                                 delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = get_plots(header, data, gconfig['lines'],
                             exclude=gconfig.get('exclude', []),
                             prefix=True)
            for key in temp.keys():
                deltas_dir['_{}'.format(key)] = temp[key].copy()

            # And here the std delta
            data = np.genfromtxt(file.format(res_dir, case, gconfig['errorfile']),
                                 delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = get_plots(header, data, gconfig['lines'],
                             exclude=gconfig.get('exclude', []),
                             prefix=True)
            for key in temp.keys():
                errors_dir['_{}'.format(key)] = temp[key].copy()

        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))

        for _, k in enumerate(plots_dir.keys()):
            values = plots_dir[k]

            # approx = k.split('approx')[1].split('_')[0]
            # approx = gconfig['approx'][approx]

            # label = '{}'.format(approx)

            x = get_values(values, header, gconfig['x_name'])
            omp = get_values(values, header, gconfig['omp_name'])

            y = get_values(values, header, gconfig['y_name'])
            ypercent = get_values(values, header, gconfig['percent_name'])
            ydelta = get_values(deltas_dir[k], header, gconfig['y_name'])
            yerr = get_values(errors_dir[k], header, gconfig['y_name'])

            # The std is normalised to ydelta
            yerr = yerr / ydelta

            # ydelta is normalized to y and then multiplied by the global_percentage
            # The results is %
            ydelta = (ydelta / y) * ypercent

            # x_new = []
            # y_new = []
            # yerr_new = []
            # for i, xi in enumerate(gconfig['x_to_keep']):
            #     if xi in x:
            #         x_new.append(xi)
            #         y_new.append(ydelta[list(x).index(xi)])
            #         yerr_new.append(yerr[list(x).index(xi)])
            # x = np.array(x_new)
            # ydelta = np.array(y_new)
            # yerr = np.array(yerr_new)

            # yerr is now denormalised
            yerr = yerr * ydelta
            yerrmin = np.minimum(yerr, ydelta)

            x = x * omp[0]

            plt.bar(np.arange(len(x)) + pos, ydelta, width=0.9*width,
                    edgecolor='0.', label=case.upper(),
                    hatch=gconfig['hatches'][col],
                    color=gconfig['colors'][col],
                    zorder=2,
                    yerr=[np.zeros(len(yerr)), yerr],
                    ecolor=gconfig['ecolor'],
                    capsize=gconfig['capsize']
                    )

            # print("{}:{}:".format(case, label), end='\t')
            # for xi, yi, yeri in zip(x//20, ydelta, yerr):
            #     print('N:{:.0f} {:.2f}±{:.2f}'.format(
            #         xi, yi, yeri), end=' ')
            # print('')
            # print("{}:{}:".format(case, label), speedup)
        pos += 1 * width
        # pos += width * step

    plt.ylim(gconfig['ylim'])
    plt.xticks(np.arange(len(x)) + step/3, np.array(x, int) //
               20, **gconfig['ticks'])

    plt.xticks(**gconfig['ticks'])
    plt.yticks(gconfig['yticks'], gconfig['yticks'], **gconfig['ticks'])
    ax.tick_params(**gconfig['tick_params_left'])

    plt.legend(**gconfig['legend'])
    plt.tight_layout()
    plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(images_dir, this_filename[:-3], '-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        fig.savefig(file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
