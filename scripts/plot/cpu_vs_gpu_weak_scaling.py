import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the weak scaling experiment.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-i', '--inputdir', type=str, default=os.path.join(project_dir, 'results'),
                    help='The directory with the results.')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='The directory to store the plots.'
                    'Default: In a plots directory inside the input results directory.')


parser.add_argument('-c', '--cases', type=str, default='lhc,sps,ps',
                    help='A comma separated list of the testcases to run. Default: lhc,sps,ps')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')
parser.add_argument('-e', '--errorbars', action='store_true',
                    help='Add errorbars.')


args = parser.parse_args()
args.cases = args.cases.split(',')

res_dir = args.inputdir
if args.outdir is None:
    images_dir = os.path.join(res_dir, 'plots')
else:
    images_dir = args.outdir

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

gconfig = {
    'approx': {
        '0': 'exact',
        '1': 'SRP',
        '2': 'RDS',
    },
    'label': {
        'exact-gpu0-tp0': 'CPU-BASE',
        'exact-gpu0-tp1': 'CPU-TP',
        'exact-gpu1-tp0': 'GPU-1PN',
        'exact-gpu2-tp0': 'GPU-2PN',

    },
    'hatches': ['', '', 'xx', 'xx'],
    'markers': ['x', 'o', '^', '+'],
    'colors': ['0.85', '0.3', '0.85', '0.3'],
    'x_name': 'n',
    # 'x_to_keep': [4, 8, 16, 32, 64],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    'xlabel': 'Nodes (x20 Cores/ x1 or x2 GPUs)',
    'ylabel': 'Norm. Throughput',
    'title': {
                # 's': '{}'.format(case.upper()),
                'fontsize': 10,
                'y': 0.83,
                # 'x': 0.55,
                'fontweight': 'bold',
    },
    'figsize': [5, 2.],
    'annotate': {
        'fontsize': 9,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center',
        'rotation': '0',
    },
    'ticks': {'fontsize': 10},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 4, 'handlelength': 1., 'fancybox': True,
        'framealpha': 0., 'fontsize': 9, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        'bbox_to_anchor': (0, 1.15),
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

    'ylim': [0., 1.1],
    # 'ylim2': [0, 110],
    'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    # 'yticks2': [0, 20, 40, 60, 80, 100],
    'outfiles': ['{}/{}-{}.png'],
    'files': [
        '{}/{}/tp-approx0-weak-scaling/comm-comp-report.csv',
        '{}/{}/approx0-weak-scaling/comm-comp-report.csv',
        '{}/{}/exact-timing-gpu/comm-comp-report.csv',
        # '{}/{}/exact-timing-gpu-512x160/comm-comp-report.csv',
        '{}/{}/approx0-weak-scaling-gpu-2pn/comm-comp-report.csv',
        # '{}/{}/approx0-weak-scaling-gpu-1pn/comm-comp-report.csv',
        # '{}/{}/approx0-weak-scaling/comm-comp-report.csv',
        # '{}/{}/lb-tp-approx2-weak-scaling/comm-comp-report.csv',
        # '{}/{}/lb-tp-approx1-weak-scaling/comm-comp-report.csv',
    ],
    'lines': {
        # 'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
        # 'lb': ['interval', 'reportonly'],
        'approx': ['0', '1', '2'],
        'gpu': ['0', '1', '2'],
        # 'lba': ['500'],
        # 'b': ['6', '12', '24', '96', '192',
        #       '48', '21', '9', '18', '36',
        #       '72', '144', '288'],
        # 't': ['5000'],
        'type': ['total'],
    }
}

# Force sans-serif math mode (for axes labels)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
plt.rcParams['font.family'] = 'sans-serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica'
# 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here

# plt.rcParams['font.family'] = gconfig['fontname']
# plt.rcParams['text.usetex'] = True


if __name__ == '__main__':
    for col, case in enumerate(args.cases):
        fig, ax = plt.subplots(ncols=1, nrows=1,
                               sharex=True, sharey=True,
                               figsize=gconfig['figsize'])
        # ax_arr = np.atleast_1d(ax_arr)
        labels = set()
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Reading data'))

        # ax = ax_arr[col]
        # ax2 = ax.twinx()
        plt.sca(ax)
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
                approx = key.split('approx')[1].split('_')[0]
                approx = gconfig['approx'][approx]
                gpu = key.split('gpu')[1].split('_')[0]
                tp = '0'
                if 'tp-approx' in file:
                    tp = '1'
                label = f'{approx}-gpu{gpu}-tp{tp}'
                label = gconfig['label'][label]
                plots_dir[label] = temp[key].copy()
                # key = '{}-gpu{}-tp{}'.format(approx, gpu, tp)
                # label = gconfig['label'][key]

        plt.grid(True, which='major', alpha=0.5)
        plt.grid(False, which='major', axis='x')
        plt.gca().set_axisbelow(True)

        plt.title('{}'.format(case.upper()), **gconfig['title'])
        # if col == 1:
        plt.xlabel(gconfig['xlabel'], labelpad=3,
                   fontweight='bold',
                   fontsize=gconfig['fontsize'])
        # if col == 0:
        plt.ylabel(gconfig['ylabel'], labelpad=3,
                   fontweight='bold',
                   fontsize=gconfig['fontsize'])

        keyref = ''
        for k in plots_dir.keys():
            if k == 'CPU-BASE':
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
        # This the reference throughput per node
        yref = parts * bunches * turns / y
        yref /= (x * omp // 20)

        # yref = yref[list(x).index(4)]

        pos = 0
        # step = 0.1
        width = 1. / (1*len(plots_dir.keys())+0.4)
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))

        for idx, k in enumerate(gconfig['label'].values()):
            if k not in plots_dir:
                continue
            values = plots_dir[k]
            # approx = k.split('approx')[1].split('_')[0]
            # approx = gconfig['approx'][approx]
            # gpu = k.split('gpu')[1].split('_')[0]
            # tp = k.split('_tp')[1]
            # key = '{}-gpu{}-tp{}'.format(approx, gpu, tp)
            label = k

            x = get_values(values, header, gconfig['x_name'])
            omp = get_values(values, header, gconfig['omp_name'])
            y = get_values(values, header, gconfig['y_name'])
            parts = get_values(values, header, 'ppb')
            bunches = get_values(values, header, 'b')
            turns = get_values(values, header, 't')

            # This is the throughput per node
            y = parts * bunches * turns / y
            y /= (x * omp//20)

            speedup = y
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
            speedup = speedup / yref
            # speedup = speedup / speedup[0]
            plt.bar(pos+np.arange(len(x)), speedup, width=0.95*width,
                    edgecolor='0', label=label, hatch=gconfig['hatches'][idx],
                    color=gconfig['colors'][idx])
            
            # if 'CPU-BASE':
            for i in np.arange(len(speedup)):
                # if speedup[i] > 0.9:
                #     continue
                ax.annotate('{:.2f}'.format(speedup[i]),
                            xy=(pos+i, speedup[i]),
                            **gconfig['annotate'])
            # plt.plot(np.arange(len(x)), speedup,
            #          label=label, marker=gconfig['markers'][idx],
            #          color=gconfig['colors'][idx])
            # print("{}:{}:".format(case, label), speedup)
            pos += 1 * width
        # pos += width * step
        # plt.xlim(0-.8*width, len(x)-1.5*width)

        plt.xticks((pos-width)/2 + np.arange(len(x)), np.array(x, int)//20, **gconfig['ticks'])
        ax.tick_params(**gconfig['tick_params_left'])
        plt.legend(**gconfig['legend'])

        # plt.ylim(gconfig['ylim'])
        # plt.yticks(gconfig['yticks'], **gconfig['ticks'])

    plt.tight_layout()
    plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(images_dir, this_filename[:-3], '-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        fig.savefig(file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
