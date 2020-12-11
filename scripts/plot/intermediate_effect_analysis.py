import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the intermediate effect analysis.',
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
        '0': '',
        '1': 'SRP',
        '2': 'RDS',
    },
    'hatches': ['', '', 'xx', '', 'xx', '', 'xx'],
    'colors': ['0.1', '0.45', '0.45', '0.7', '0.7', '0.95', '0.95'],
    'x_name': 'n',
    # 'x_to_keep': [16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    'xlabel': 'Nodes (x20 Cores)',
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
    'xticks': {'fontsize': 10, 'rotation': '0', 'fontweight': 'bold'},
    'ticks': {'fontsize': 10, 'rotation': '0'},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 7, 'handlelength': 1.1, 'fancybox': True,
        'framealpha': 0., 'fontsize': 9, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.3,
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
    'ylim': [0.45, 1.02],
    # 'ylim2': [10, 90],
    'yticks': [0.5, 0.6, 0.7, .8, .9, 1.],
    # 'yticks2': [0, 20, 40, 60, 80, 100],
    'outfiles': ['{}/{}-{}.png'],
    'files': [
        '{}/{}/approx0-interm/comm-comp-report.csv',
        '{}/{}/approx2-interm/comm-comp-report.csv',
        '{}/{}/approx1-interm/comm-comp-report.csv',
        '{}/{}/tp-approx0-interm/comm-comp-report.csv',
        '{}/{}/lb-tp-approx0-interm/comm-comp-report.csv',
        '{}/{}/lb-tp-approx2-interm/comm-comp-report.csv',
        '{}/{}/lb-tp-approx1-interm/comm-comp-report.csv',
    ],
    'lines': {
        # 'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
        'lb': ['interval', 'reportonly'],
        'approx': ['0', '1', '2'],
        # 'lba': ['500'],
        # 'b': ['96', '48', '72', '21'],
        # 't': ['5000'],
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
    plt.ylabel(gconfig['ylabel'], labelpad=3,
               fontweight='bold',
               fontsize=gconfig['fontsize'])
    pos = 0
    step = 1.
    labels = set()
    avg = {}
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
                if 'tp-approx' in file:
                    plots_dir['_{}_tp1'.format(key)] = temp[key].copy()
                else:
                    plots_dir['_{}_tp0'.format(key)] = temp[key].copy()
        width = .95*step / (len(plots_dir.keys()))

        # First the reference value
        keyref = ''
        for k in plots_dir.keys():
            if 'approx0' in k and 'tp0' in k and 'lbreportonly' in k:
                keyref = k
                break
        if keyref == '':
            print('ERROR: reference key not found')
            exit(-1)

        refvals = plots_dir[keyref]

        x = get_values(refvals, header, gconfig['x_name'])
        omp = get_values(refvals, header, gconfig['omp_name'])
        y = get_values(refvals, header, gconfig['y_name'])
        parts = get_values(refvals, header, 'ppb')
        bunches = get_values(refvals, header, 'b')
        turns = get_values(refvals, header, 't')
        yref = parts * bunches * turns / y

        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))

        for idx, k in enumerate(plots_dir.keys()):
            values = plots_dir[k]
            # mpiv = k.split('_mpi')[1].split('_')[0]
            lb = k.split('lb')[1].split('_')[0]
            approx = k.split('approx')[1].split('_')[0]
            tp = k.split('_')[-1]
            experiment = k.split('_')[-1]
            if lb == 'interval':
                lb = 'LB-'
            elif lb == 'reportonly':
                lb = ''
            if tp == 'tp1':
                tp = 'TP-'
            elif tp == 'tp0':
                tp = ''
            approx = gconfig['approx'][approx]

            label = '{}{}{}'.format(lb, tp, approx)
            if label == '':
                label = 'base'
            if label[-1] == '-':
                label = label[:-1]

            x = get_values(values, header, gconfig['x_name'])
            omp = get_values(values, header, gconfig['omp_name'])
            y = get_values(values, header, gconfig['y_name'])
            parts = get_values(values, header, 'ppb')
            bunches = get_values(values, header, 'b')
            turns = get_values(values, header, 't')

            # This is the throughput
            y = parts * bunches * turns / y

            speedup = yref / y
            # x_new = []
            # sp_new = []
            # for i, xi in enumerate(gconfig['x_to_keep']):
            #     x_new.append(xi)
            #     if xi in x:
            #         sp_new.append(speedup[list(x).index(xi)])
            #     else:
            #         sp_new.append(0)
            # x = np.array(x_new)
            # speedup = np.array(sp_new)
            # efficiency = 100 * speedup / (x * omp[0] / ompref)
            x = x * omp[0]

            # width = .9 * step / (len(x))
            if label not in avg:
                avg[label] = []
            avg[label].append(speedup)
            # efficiency = 100 * speedup / x
            if label in labels:
                label = None
            else:
                labels.add(label)
            plt.bar(pos + idx*width, speedup, width=0.9*width,
                    edgecolor='0.', label=label, hatch=gconfig['hatches'][idx],
                    color=gconfig['colors'][idx])
        pos += step

    # vals = np.mean(avg, axis=0)
    for idx, key in enumerate(avg.keys()):
        vals = avg[key]
        val = np.mean(vals)
        plt.bar(pos + idx*width, val, width=0.9*width,
                edgecolor='0.', label=None, hatch=gconfig['hatches'][idx],
                color=gconfig['colors'][idx])
        text = '{:.2f}'.format(val)
        if idx == 0:
            text = ''
        else:
            text = text[:]
        ax.annotate(text, xy=(pos + idx*width, 0.01 + val),
                    rotation='90',
                    **gconfig['annotate'])
    pos += step

    plt.ylim(gconfig['ylim'])
    handles, labels = ax.get_legend_handles_labels()
    # print(labels)

    plt.legend(handles=handles, labels=labels, **gconfig['legend'])
    plt.xlim(0-1.3*width/2, pos-1.4*width/2)
    plt.yticks(gconfig['yticks'], **gconfig['ticks'])

    plt.xticks(np.arange(pos) + step/2,
               [c.upper() for c in args.cases] + ['AVG'], **gconfig['xticks'])

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
