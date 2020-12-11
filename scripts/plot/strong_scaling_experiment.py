import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure with of the strong scaling experiment.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-i', '--inputdir', type=str, default=os.path.join(project_dir, 'results'),
                    help='The directory with the results.')

parser.add_argument('-c', '--cases', type=str, default='lhc,sps,ps',
                    help='A comma separated list of the testcases to run. Default: lhc,sps,ps')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')

parser.add_argument('-ne', '--no-errorbars', action='store_true',
                    help='Add errorbars.')


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
    'hatches': ['', '', 'xx'],
    'markers': ['x', 'o', '^'],
    'colors': ['xkcd:red', 'xkcd:green', 'xkcd:blue'],
    'x_name': 'n',
    # 'x_to_keep': [4, 8, 16, 32, 64],
    # 'x_to_keep': [8, 16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    # 'y_err_name': 'std',
    'xlabel': 'Nodes (x20 Cores)',
    'ylabel': 'Speedup',
    'title': {
                # 's': '{}'.format(case.upper()),
                'fontsize': 10,
                'y': .85,
                # 'x': 0.55,
                'fontweight': 'bold',
    },
    'figsize': [5, 2.2],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 10},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 1, 'handlelength': 1.5, 'fancybox': False,
        'framealpha': .7, 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        'bbox_to_anchor': (0., 0.85)
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
    'ylim': [0, 36],
    'xlim': [1.6, 36],
    'yticks': [4, 8, 12, 16, 20, 24, 28, 32],
    'outfiles': ['{}/{}-{}.png'],
    'errorfile': 'comm-comp-std-report.csv',
    'datafile': 'comm-comp-report.csv',
    'files': [
        '{}/{}/lb-tp-approx0-strong-scaling/{}',
        '{}/{}/lb-tp-approx2-strong-scaling/{}',
        '{}/{}/lb-tp-approx1-strong-scaling/{}',
    ],
    'lines': {
        'approx': ['0', '1', '2'],
        'type': ['total'],
    },
    'reference': {
        'file': '{}/baseline/{}/strong-scaling/comm-comp-report.csv',
        'lines': {
            'b': ['192', '21', '288'],
            'ppb': ['4000000', '16000000'],
            # 't': ['5000'],
            'type': ['total'],
        },
    }
}

plt.rcParams['font.family'] = gconfig['fontname']
# plt.rcParams['text.usetex'] = True

if __name__ == '__main__':
    fig, ax_arr = plt.subplots(ncols=len(args.cases), nrows=1,
                               sharex=True, sharey=True,
                               figsize=gconfig['figsize'])
    ax_arr = np.atleast_1d(ax_arr)
    labels = set()
    for col, case in enumerate(args.cases):
        print('[{}] tc: {}: {}'.format(this_filename[:-3], case, 'Reading data'))

        ax = ax_arr[col]
        plt.sca(ax)
        ax.set_xscale('log', basex=2)
        plots_dir = {}
        errors_dir = {}

        for file in gconfig['files']:
            # print(file)
            data = np.genfromtxt(file.format(res_dir, case, gconfig['datafile']),
                                 delimiter='\t', dtype=str)
            header, data = list(data[0]), data[1:]
            temp = get_plots(header, data, gconfig['lines'],
                             exclude=gconfig.get('exclude', []),
                             prefix=True)
            for key in temp.keys():
                plots_dir['_{}_'.format(key)] = temp[key].copy()

            if not args.no_errorbars:
                data = np.genfromtxt(file.format(res_dir, case, gconfig['errorfile']),
                                     delimiter='\t', dtype=str)
                header, data = list(data[0]), data[1:]
                temp = get_plots(header, data, gconfig['lines'],
                                 exclude=gconfig.get('exclude', []),
                                 prefix=True)
                for key in temp.keys():
                    errors_dir['_{}_'.format(key)] = temp[key].copy()
        ref_dir = {}
        data = np.genfromtxt(gconfig['reference']['file'].format(res_dir, case),
                             delimiter='\t', dtype=str)
        header, data = list(data[0]), data[1:]
        temp = get_plots(header, data, gconfig['reference']['lines'],
                         exclude=gconfig.get('exclude', []),
                         prefix=True)
        for key in temp.keys():
            ref_dir[case] = temp[key].copy()


        plt.grid(True, which='both', axis='y', alpha=0.5)
        # plt.grid(True, which='minor', alpha=0.5, zorder=1)
        plt.grid(False, which='major', axis='x')
        plt.title('{}'.format(case.upper()), **gconfig['title'])
        if col == 1:
            plt.xlabel(gconfig['xlabel'], labelpad=3,
                       fontweight='bold',
                       fontsize=gconfig['fontsize'])
        if col == 0:
            plt.ylabel(gconfig['ylabel'], labelpad=3,
                       fontweight='bold',
                       fontsize=gconfig['fontsize'])

        pos = 0
        step = 0.1
        width = 1. / (1*len(plots_dir.keys())+0.4)

        print('[{}] tc: {}: {}'.format(this_filename[:-3], case, 'Plotting data'))

        for idx, k in enumerate(plots_dir.keys()):
            values = plots_dir[k]
            approx = k.split('approx')[1].split('_')[0]
            experiment = k.split('_')[-1]
            approx = gconfig['approx'][approx]

            label = '{}'.format(approx)

            x = get_values(values, header, gconfig['x_name'])
            omp = get_values(values, header, gconfig['omp_name'])
            y = get_values(values, header, gconfig['y_name'])
            parts = get_values(values, header, 'ppb')
            bunches = get_values(values, header, 'b')
            turns = get_values(values, header, 't')
            if not args.no_errorbars:
                # yerr is normalized to y
                yerr = get_values(errors_dir[k], header, gconfig['y_name'])
                yerr = yerr/y
            else:
                yerr = np.zeros(len(y))

            # This is the throughput
            y = parts * bunches * turns / y

            # Now the reference, 1thread
            yref = get_values(ref_dir[case], header, gconfig['y_name'])
            partsref = get_values(ref_dir[case], header, 'ppb')
            bunchesref = get_values(ref_dir[case], header, 'b')
            turnsref = get_values(ref_dir[case], header, 't')
            ompref = get_values(ref_dir[case], header, gconfig['omp_name'])
            yref = partsref * bunchesref * turnsref / yref

            speedup = y / yref

            # x_new = []
            # sp_new = []
            # yerr_new = []
            # for i, xi in enumerate(gconfig['x_to_keep']):
            #     if xi in x:
            #         x_new.append(xi)
            #         sp_new.append(speedup[list(x).index(xi)])
            #         yerr_new.append(yerr[list(x).index(xi)])
            #     # else:
            #     #     sp_new.append(0)
            # x = np.array(x_new)
            # speedup = np.array(sp_new)
            # yerr = np.array(yerr_new)
            # yerr is denormalized again
            yerr = yerr * speedup
            # efficiency = 100 * speedup / (x * omp[0] / ompref)
            x = x * omp[0]

            plt.errorbar(x//20, speedup,
                         label=label, marker=gconfig['markers'][idx],
                         color=gconfig['colors'][idx],
                         yerr=yerr,
                         capsize=2)
            # print("{}:{}:".format(case, label), end='\t')
            # for xi, yi, yeri in zip(x//20, speedup, yerr):
            #     print('N:{:.0f} {:.2f}±{:.2f}'.format(
            #         xi, yi, yeri), end=' ')
            # print('')
            # print("{}:{}:".format(case, label), speedup)
            pos += 1 * width
        # pos += width * step
        plt.ylim(gconfig['ylim'])
        # plt.xticks(np.arange(len(x)), np.array(x, int)//20)
        plt.xlim(gconfig['xlim'])
        plt.xticks(x//20, np.array(x, int)//20, **gconfig['ticks'])

        if col == 0:
            ax.tick_params(**gconfig['tick_params_left'])
        else:
            ax.tick_params(**gconfig['tick_params_center_right'])

        # if col == 0:
            # handles, labels = ax.get_legend_handles_labels()
            # print(labels)
        ax.legend(**gconfig['legend'])

        plt.xticks(**gconfig['ticks'])
        plt.yticks(gconfig['yticks'], gconfig['yticks'], **gconfig['ticks'])

    # plt.legend(**gconfig['legend'])
    plt.tight_layout()
    plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(images_dir, this_filename[:-3], '-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))

        fig.savefig(file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
