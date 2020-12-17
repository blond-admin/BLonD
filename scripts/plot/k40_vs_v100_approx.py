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

parser.add_argument('-i', '--inputdirs', type=str, nargs=3,
                    help='The input directories (gpu1, gpu2, cpu).')

parser.add_argument('-m', '--models', type=str, nargs=3, default=['k40', 'v100', 'xeon'],
                    help='The platform names.')

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

assert len(args.models) == len(args.inputdirs)
assert len(args.models) == 3

if args.outdir is None:
    images_dir = os.path.join(args.inputdirs[1], 'plots')
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
    # 'label': {
    #     'exact-gpu0-tp0': 'CPU-BASE',
    #     'exact-gpu0-tp1': 'CPU-TP',
    #     'exact-gpu1-tp0': 'GPU-1PN',
    #     'exact-gpu2-tp0': 'GPU-2PN',
    # },

    # 'hatches': ['', '', 'xx', 'xx'],
    # 'markers': ['x', 'o', '^', '+'],
    # 'colors': ['0.85', '0.3', '0.85', '0.3'],
    'label': {
        'double-exact-gpu0': 'f64-cpu',
        'tp-double-exact-gpu0': 'f64-cpu-tp',
        # 'double-exact-gpu0': 'base',
        'double-exact-gpu1': 'f64-gpu',
        'single-exact-gpu1': 'f32-gpu',
        # 'single-SRP-2-gpu1': 'f32-SRP-2',
        'double-SRP-3-gpu1': 'f64-SRP-3-gpu',
        'single-SRP-3-gpu1': 'f32-SRP-3-gpu',
        # 'single-RDS-gpu1': 'f32-RDS-gpu',
        # 'double-SRP-2-gpu1': 'f64-SRP-2',
        # 'double-RDS-gpu1': 'f64-RDS',
    },
    # 'hatches': ['', '', 'xx', '', 'xx', '', 'xx', '', 'xx'],
    'colors': ['xkcd:black', 'xkcd:red', 'xkcd:blue'],
    # 'colors': {
    #     'f64-cpu': 'xkcd:black',
    #     'f64-cpu-tp': 'xkcd:black',
    #     'f64-gpu': 'xkcd:blue',
    #     'f32-gpu': 'xkcd:blue',
    #     'f32-SRP-2-gpu': 'xkcd:red',
    #     'f32-SRP-3-gpu': 'xkcd:red',
    #     'f64-SRP-2-gpu': 'xkcd:orange',
    #     'f64-SRP-3-gpu': 'xkcd:orange',
    #     'f32-RDS-gpu': 'xkcd:purple',
    #     'f64-RDS-gpu': 'xkcd:purple',
    # },
    'hatches': {
        'f64-cpu': '',
        'f64-gpu': '',
        'f32-gpu': 'xx',
        'f32-SRP-3-gpu': '//',
        'f64-SRP-3-gpu': '\\\\',

        # 'f64-cpu-tp': 'xx',
        # 'f32-SRP-2-gpu': 'x',
        # 'f64-SRP-2-gpu': 'x',
        # 'f64-SRP-3-gpu': 'o',
        # 'f64-RDS-gpu': 'o',
    },
    'x_name': 'n',
    # 'x_to_keep': [4, 8, 16, 32, 64],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    'xlabel': 'Platform/ Version',
    'ylabel': 'Speedup',
    'cores_per_node': 20,
    'title': {
                # 's': '{}: {} vs {} vs {}',
                'fontsize': 10,
                # 'y': 0.83,
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
    'xticks': {'rotation': 20, 'fontsize': 9},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 1, 'handlelength': 1.5, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        # 'bbox_to_anchor': (0, 1.15),
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
    'gpu_files_conf': {
        # '{}/{}/tp-approx0-weak-scaling/comm-comp-report.csv',
        # '{}/{}/approx0-weak-scaling/comm-comp-report.csv',
        'files': [
            '{}/{}/exact-timing-gpu/comm-comp-report.csv',
            '{}/{}/float32-timing-gpu/comm-comp-report.csv',
            '{}/{}/f32-rds-timing-gpu/comm-comp-report.csv',
            '{}/{}/rds-timing-gpu/comm-comp-report.csv',
            '{}/{}/f32-srp-timing-gpu/comm-comp-report.csv',
            '{}/{}/srp-timing-gpu/comm-comp-report.csv',
        ],
        'lines': {
            # 'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
            # 'lb': ['interval', 'reportonly'],
            'approx': ['0', '1', '2'],
            'gpu': ['0', '1', '2'],
            'red': ['1', '2', '3'],
            'prec': ['single', 'double'],
            # 'lba': ['500'],
            # 'b': ['6', '12', '24', '96', '192',
            #       '48', '21', '9', '18', '36',
            #       '72', '144', '288'],
            # 't': ['5000'],
            'type': ['total'],
        }

        # '{}/{}/exact-timing-gpu-512x160/comm-comp-report.csv',
        # '{}/{}/approx0-weak-scaling-gpu-2pn/comm-comp-report.csv',
        # '{}/{}/approx0-weak-scaling-gpu-1pn/comm-comp-report.csv',
        # '{}/{}/approx0-weak-scaling/comm-comp-report.csv',
        # '{}/{}/lb-tp-approx2-weak-scaling/comm-comp-report.csv',
        # '{}/{}/lb-tp-approx1-weak-scaling/comm-comp-report.csv',
    },
    'cpu_files_conf': {
        'files': ['{}/{}/cpu-baseline/comm-comp-report.csv'],
        'lines': {
            'b': ['12', '21', '18'],
            'ppb': ['1000000', '1500000'],
            'approx': ['0', '1', '2'],
            'gpu': ['0', '1', '2'],
            'red': ['1', '2', '3'],
            'prec': ['single', 'double'],
            # 't': ['5000'],
            'type': ['total'],
        },
    },

}

# Force sans-serif math mode (for axes labels)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}']
plt.rcParams['font.family'] = 'DejaVu Sans Mono'  # ... for regular text
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

        cpu_model = args.models[-1]
        base_gpu_model = args.models[0]
        gpu_model = args.models[1]

        for res_dir, model in zip(args.inputdirs, args.models):
            if model not in plots_dir:
                plots_dir[model] = {}
            if model == cpu_model:
                files_conf = gconfig['cpu_files_conf']
            else:
                files_conf = gconfig['gpu_files_conf']

            for file in files_conf['files']:
                file = file.format(res_dir, case)

                # print(file)
                data = np.genfromtxt(file, delimiter='\t', dtype=str)
                header, data = list(data[0]), data[1:]
                temp = get_plots(header, data, files_conf['lines'],
                                 exclude=files_conf.get('exclude', []),
                                 prefix=True)

                for key in temp.keys():
                    approx = key.split('approx')[1].split('_')[0]
                    approx = gconfig['approx'][approx]
                    red = key.split('red')[1].split('_')[0]
                    prec = key.split('prec')[1].split('_')[0]
                    gpu = key.split('gpu')[1].split('_')[0]
                    if approx == 'SRP':
                        label = f'{prec}-{approx}-{red}-gpu{gpu}'
                    else:
                        label = f'{prec}-{approx}-gpu{gpu}'
                    if 'tp-approx' in file:
                        label = 'tp-' + label
                    if label not in gconfig['label']:
                        continue
                    label = gconfig['label'][label]
                    plots_dir[model][label] = temp[key].copy()

        keyref = ''
        for k in plots_dir[cpu_model].keys():
            if k == 'f64-cpu':
                keyref = k
                break
        if keyref == '':
            print('ERROR: reference key not found')
            exit(-1)
        refvals = plots_dir[cpu_model][keyref]
        x = get_values(refvals, header, gconfig['x_name'])
        omp = get_values(refvals, header, gconfig['omp_name'])
        y = get_values(refvals, header, gconfig['y_name'])
        parts = get_values(refvals, header, 'ppb')
        bunches = get_values(refvals, header, 'b')
        turns = get_values(refvals, header, 't')
        # This the reference throughput per node
        yref = parts * bunches * turns / y
        yref /= (x * omp // gconfig['cores_per_node'])

        # yref = yref[list(x).index(4)]

        pos = 0
        # step = 0.1
        width = 1.
        # width = 1. / (1*len(plots_dir.keys())+0.4)
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))
        # xticks_edges=[]
        xticks = []
        xtickspos = []
        labels = set()
        for model, color in zip([cpu_model, base_gpu_model, gpu_model],
                                gconfig['colors']):

            # xticks_edges.append((pos, 0))
            # dic = plots_dir[model]
            for idx, k in enumerate(gconfig['label'].values()):
                if k not in plots_dir[model]:
                    continue
                values = plots_dir[model][k]
                if model == cpu_model:
                    label = k.replace('-cpu', '')
                else:
                    label = k.replace('-gpu', '')
                x = get_values(values, header, gconfig['x_name'])
                omp = get_values(values, header, gconfig['omp_name'])
                y = get_values(values, header, gconfig['y_name'])
                parts = get_values(values, header, 'ppb')
                bunches = get_values(values, header, 'b')
                turns = get_values(values, header, 't')

                # This is the throughput per node
                y = parts * bunches * turns / y
                # y /= (x * omp//gconfig['cores_per_node'])
                y /= x
                speedup = (y / yref)[0]
                x = (x * omp[0])[0]

                if model in labels:
                    legend = None
                else:
                    legend = model
                    labels.add(model)

                plt.bar(pos, speedup, width=0.95*width,
                        edgecolor=color, label=legend,
                        hatch=gconfig['hatches'][k],
                        color='0.75')

                ax.annotate('{:.2f}'.format(speedup),
                            xy=(pos, speedup),
                            **gconfig['annotate'])
                xtickspos.append(pos)
                xticks.append(label)
                pos += width
            pos += 0.5 * width
            # xticks_edges[-1][1] = pos

        plt.grid(True, which='major', alpha=0.5)
        plt.grid(False, which='major', axis='x')
        plt.gca().set_axisbelow(True)

        plt.title(f'{case.upper()}: {cpu_model.upper()} vs {base_gpu_model.upper()} vs {gpu_model.upper()}',
                  **gconfig['title'])
        # if col == 1:
        # plt.xlabel(gconfig['xlabel'], labelpad=3,
        #            fontweight='bold',
        #            fontsize=gconfig['fontsize'])
        # if col == 0:
        plt.ylabel(gconfig['ylabel'], labelpad=3,
                   fontweight='bold',
                   fontsize=gconfig['fontsize'])

        plt.xticks(xtickspos, xticks, **gconfig['xticks'])
        ax.tick_params(**gconfig['tick_params_left'])
        plt.legend(**gconfig['legend'])
        ylims = plt.gca().get_ylim()
        plt.ylim(ymax=ylims[1]+2)
        # plt.ylim(gconfig['ylim'])
        # plt.yticks(gconfig['yticks'], **gconfig['ticks'])

        plt.tight_layout()
        # plt.subplots_adjust(**gconfig['subplots_adjust'])
        for file in gconfig['outfiles']:
            file = file.format(images_dir, this_filename[:-3], case)
            print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
            fig.savefig(file, dpi=600, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
