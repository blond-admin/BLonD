import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse


# python scripts/plot/approx_strong_scaling_cpu_gpu_2pn.py -icpu results/weak-scaling-cpu/ -igpu results/aris2/ -igpu2 results/aris2/ -b results/baselinecpu/ -o results/multiplatform/plots -s

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the intermediate effect analysis.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-icpu', '--inputcpu', type=str, default=None,
                    help='The directory with the CPU results.')

parser.add_argument('-igpu', '--inputgpu', type=str, default=None,
                    help='The directory with the GPU results.')


parser.add_argument('-igpu2', '--inputgpu2', type=str, default=None,
                    help='The directory with the GPU 2pn results.')


parser.add_argument('-b', '--basedir', type=str, default=None,
                    help='The directory with the baseline results.')


parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='The directory to store the plots.'
                    'Default: In a plots directory inside the input results directory.')

parser.add_argument('-c', '--cases', type=str, default='lhc,sps,ps',
                    help='A comma separated list of the testcases to run. Default: lhc,sps,ps')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')


args = parser.parse_args()
args.cases = args.cases.split(',')

res_dir = args.inputcpu
if args.outdir is None:
    images_dir = os.path.join(this_directory, 'temp')
else:
    images_dir = args.outdir

if args.basedir is None:
    args.basedir = args.inputcpu

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

gconfig = {
    'approx': {
        '0': '',
        '1': 'SRP',
        '2': 'RDS',
    },
    'label': {
        'doublecpu': 'HBLonD',
        'singleSRPcpu': 'HBLonD-F32-SRP',
        'singleRDScpu': 'HBLonD-F32-RDS',

        'doublegpu': 'CuBLonD-1',
        'singleSRPgpu': 'CuBLonD-1-F32-SRP',
        'singleRDSgpu': 'CuBLonD-1-F32-RDS',

        'doublegpu2': 'CuBLonD-2',
        'singleSRPgpu2': 'CuBLonD-2-F32-SRP',
        'singleRDSgpu2': 'CuBLonD-2-F32-RDS',


    },
    'legends': {
        'HBLonD': 'HBLonD',
        'HBLonD-F32-SRP': 'F32-SRP',
        'HBLonD-F32-RDS': 'F32-RDS',

        'CuBLonD-1': 'CuBLonD-1GPU',
        'CuBLonD-1-F32-SRP': 'F32-SRP',
        'CuBLonD-1-F32-RDS': 'F32-RDS',

        'CuBLonD-2': 'CuBLonD-2GPU',
        'CuBLonD-2-F32-SRP': 'F32-SRP',
        'CuBLonD-2-F32-RDS': 'F32-RDS',


    },
    'colors': {
        'HBLonD': '0.85',
        'HBLonD-F32-SRP': '0.6',
        'HBLonD-F32-RDS': '0.3',

        'CuBLonD-1': '0.85',
        'CuBLonD-1-F32-SRP': '0.6',
        'CuBLonD-1-F32-RDS': '0.3',

        'CuBLonD-2': '0.85',
        'CuBLonD-2-F32-SRP': '0.6',
        'CuBLonD-2-F32-RDS': '0.3',

    },

    'hatches': {
        'CuBLonD-1': '',
        'CuBLonD-1-F32-SRP': '///',
        'CuBLonD-1-F32-RDS': '///',

        'CuBLonD-2': '',
        'CuBLonD-2-F32-SRP': '///',
        'CuBLonD-2-F32-RDS': '///',
        
        'HBLonD': '',
        'HBLonD-F32-SRP': '///',
        'HBLonD-F32-RDS': '///',
    },


    'x_name': 'N',
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    'xlabel': {
        'xlabel': 'Nodes'
    },
    'ylabel': 'Speedup',
    'title': {
        # 's': '',
        'fontsize': 11,
        'y': 0.8,
        # 'x': 0.8,
        'fontweight': 'bold',
    },
    'figsize': [5, 5],
    'annotate': {
        'fontsize': 10,
        'textcoords': 'data',
        'va': 'top',
        # 'ha': 'right'
    },
    'xticks': {'fontsize': 10, 'rotation': '0'},
    'ticks': {'fontsize': 10, 'rotation': '0'},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 1, 'handlelength': 1.3, 'fancybox': True,
        'framealpha': .5, 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.3,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.3,
        # 'bbox_to_anchor': (-0.04, 1.16)
    },
    'subplots_adjust': {
        'wspace': 0.05, 'hspace': 0.12, 'top': .95
    },
    'tick_params': {
        'pad': 2, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'inout', 'length': 2, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    'xlim': [-0.37025, 4.97025],
    'cpu': {
        'ylim': [0, 60],
        # 'ylim': [0,100],
        'yticks': [0, 10, 20, 30, 40, 50, 60],
        # 'yticks': [0, 20, 40, 60, 80, 100]
    },
    'gpu':{
        'ylim': [0,90],
        # 'yticks': [0, 20, 40, 60, 80, 100]
        'yticks': [0, 15, 30, 45, 60, 75, 90]

    },
    'gpu2':{
        'ylim': [0,90],
        'yticks': [0, 15, 30, 45, 60, 75, 90]
        # 'yticks': [0, 20, 40, 60, 80, 100]
    },

    # 'ylim': [0, 60],
    # 'yticks': [0, 10, 20, 30, 40, 50, 60],
    'outfiles': [
        '{}/{}-{}-nodes-{}.png',
        '{}/{}-{}-nodes-{}.pdf'
    ],
    'files-cpu': [
        '{}/{}/exact-timing-cpu/comm-comp-report.csv',
        '{}/{}/f32-rds-timing-cpu/comm-comp-report.csv',
        '{}/{}/f32-srp-timing-cpu/comm-comp-report.csv',

    ],
    'files-gpu': [
        '{}/{}/exact-timing-strong-gpu/comm-comp-report.csv',
        '{}/{}/f32-rds-timing-strong-gpu/comm-comp-report.csv',
        '{}/{}/f32-srp-timing-strong-gpu/comm-comp-report.csv',
    ],

    'files-gpu2': [
        '{}/{}/exact-timing-strong-gpu-2pn/comm-comp-report.csv',
        '{}/{}/f32-rds-timing-strong-gpu-2pn/comm-comp-report.csv',
        '{}/{}/f32-srp-timing-strong-gpu-2pn/comm-comp-report.csv',
    ],

    'lines-cpu': {
        'approx': ['0', '1', '2'],
        'red': ['1', '3'],
        'prec': ['single', 'double'],
        'omp': ['10', '20'],
        'gpu': ['0'],
        # 'ppb': ['3000000', '2000000', '32000000'],
        # 'b': ['192', '288', '21'],
        'type': ['total'],
    },

    'lines-gpu': {
        'approx': ['0', '1', '2'],
        'red': ['1', '3'],
        'prec': ['single', 'double'],
        'omp': ['10', '20'],
        'gpu': ['1'],
        'ppb': ['4000000', '2000000', '32000000'],
        'b': ['192', '288', '21'],
        'type': ['total'],
    },

    'lines-gpu2': {
        'approx': ['0', '1', '2'],
        'red': ['1', '3'],
        'prec': ['single', 'double'],
        'omp': ['10', '20'],
        'gpu': ['2'],
        'ppb': ['4000000', '2000000', '32000000'],
        'b': ['192', '288', '21'],
        'type': ['total'],
    },

    'linefilter-cpu': [
        {'N': ['1', '20']},
    ],
    'linefilter-gpu': [
        {'N': ['1', '32']},
    ],
    'linefilter-gpu2': [
        {'N': ['1', '32']},
    ],
    'reference': {
        'file': '{}/{}/comm-comp-report.csv',
        'lines': {
            # 'b': ['12', '21', '18'],
            # 'b': ['192', '288', '21'],
            'b': ['12', '18', '21'],
            'ppb': ['1500000', '1000000'],
            # 't': ['5000'],
            'type': ['total'],
        },
    }

}

# Force sans-serif math mode (for axes labels)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}'
plt.rcParams['font.family'] = gconfig['fontname']


if __name__ == '__main__':

    fig, ax_arr = plt.subplots(ncols=len(args.cases), nrows=3,
                               sharex=True, sharey=False,
                               figsize=gconfig['figsize'])
    # pos = 0
    step = 1.
    labels = set()
    xticks = []
    xtickspos = []
    for col, case in enumerate(args.cases):
        # ax = ax_arr[col]
        # plt.sca(ax)

        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Reading data'))
        plots_dir = {}
        for row, indir, platform in zip([0, 1, 2], [args.inputcpu, args.inputgpu, args.inputgpu2],
                                        ['cpu', 'gpu', 'gpu2']):
            if not indir:
                continue
            plots_dir[platform] = {}
            for file in gconfig['files-{}'.format(platform)]:
                file = file.format(indir, case)
                # print(file)
                data = np.genfromtxt(file, delimiter='\t', dtype=str)
                header, data = list(data[0]), data[1:]
                temp = get_plots(header, data, gconfig['lines-{}'.format(platform)],
                                 exclude=gconfig.get('exclude', []),
                                 linefilter=gconfig.get(f'linefilter-{platform}', {}),
                                 prefix=True)
                for key in temp.keys():
                    if 'tp-approx' in file:
                        plots_dir[platform]['_{}_tp1'.format(
                            key)] = temp[key].copy()
                    else:
                        plots_dir[platform]['_{}_tp0'.format(
                            key)] = temp[key].copy()

        data = np.genfromtxt(gconfig['reference']['file'].format(args.basedir, case),
                             delimiter='\t', dtype=str)
        header, data = list(data[0]), data[1:]
        ref_dir = get_plots(header, data, gconfig['reference']['lines'],
                            exclude=gconfig.get('exclude', []),
                            prefix=True)

        print('Ref keys: ', ref_dir.keys())
        ref_dir_keys = list(ref_dir.keys())
        # assert len(ref_dir_keys) == 1
        refvals = ref_dir[ref_dir_keys[-1]]

        xref = get_values(refvals, header, gconfig['x_name'])
        omp = get_values(refvals, header, gconfig['omp_name'])
        y = get_values(refvals, header, gconfig['y_name'])
        parts = get_values(refvals, header, 'ppb')
        bunches = get_values(refvals, header, 'b')
        turns = get_values(refvals, header, 't')
        yref = parts * bunches * turns / y

        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))
        # To sort the keys, by approx and then reduce value
        # print(plots_dir[platform].keys())
        # keys = ['_'.join(a.split('_')[1:4]) for a in list(plots_dir[platform].keys())]
        # print(keys)
        # keys = np.array(list(plots_dir[platform].keys()))[np.argsort(keys)]
        for row, platform in enumerate(['cpu', 'gpu', 'gpu2']):
            ax = ax_arr[row][col]
            plt.sca(ax)

            width = .9 * step / (len(plots_dir[platform].keys()))
            for idx, k in enumerate(plots_dir[platform].keys()):
                values = plots_dir[platform][k]
                approx = k.split('approx')[1].split('_')[0]
                red = k.split('red')[1].split('_')[0]
                experiment = k.split('_')[-1]
                prec = k.split('prec')[1].split('_')[0]
                # gpus = k.split('gpu')[1].split('_')[0]

                approx = gconfig['approx'][approx]
                label = gconfig['label'][prec+approx+platform]

                x = get_values(values, header, gconfig['x_name'])
                # omp = get_values(values, header, gconfig['omp_name'])
                y = get_values(values, header, gconfig['y_name'])
                parts = get_values(values, header, 'ppb')
                bunches = get_values(values, header, 'b')
                turns = get_values(values, header, 't')

                # This is the throughput
                y = parts * bunches * turns / y

                speedup = y / yref
                legend = label
                if label in labels:
                    legend = None
                else:
                    labels.add(label)

                print("{}:{}:{}:".format(
                    platform, case, label))
                print(x)
                print(speedup)

                xpos = idx * width + np.arange(len(x))
                plt.bar(xpos, speedup,
                        edgecolor='black',
                        width=0.85 * width,
                        label=gconfig['legends'].get(legend, None),
                        hatch=gconfig['hatches'][label],
                        color=gconfig['colors'][label])
                for xi, yi in zip(xpos, speedup):
                    if yi > gconfig[platform]['ylim'][1]:
                        if idx == 1:
                            ha = 'right'
                            offset = -0.2
                        else:
                            ha = 'left'
                            offset = 0.2
                        plt.gca().annotate('{:.0f}'.format(yi),
                                           xy=(xi+offset, gconfig[platform]['ylim'][1]-1),
                                           ha=ha,
                                           **gconfig['annotate'])
            plt.grid(True, which='major', alpha=0.5)
            plt.grid(False, which='major', axis='x')
            plt.gca().set_axisbelow(True)
            # plt.gca().set_yscale('log', base=10)
            if row == 0 and col == 0:
                plt.title(r'\textbf{'+'{}'.format(case.upper())+'}', x=0.8, **gconfig['title'])
            elif row == 0 and col > 0:
                plt.title(r'\textbf{'+'{}'.format(case.upper())+'}', x=0.5, **gconfig['title'])

            if col == 0:
                handles, labs = ax.get_legend_handles_labels()
                plt.ylabel(gconfig['ylabel'], labelpad=1)
                plt.legend(handles=handles, labels=labs, **gconfig['legend'])

            if col == 0:
                plt.ylim(gconfig[platform]['ylim'])
                plt.yticks(gconfig[platform]['yticks'], **gconfig['ticks'])
            else:
                plt.ylim(gconfig[platform]['ylim'])
                plt.yticks(gconfig[platform]['yticks'], fontsize=0)

            plt.xticks(xtickspos, np.array(xticks, int), **gconfig['xticks'])
            plt.xlim(gconfig['xlim'])

            if row == 2 and col == 1:
                plt.xlabel(**gconfig['xlabel'])

            ax.tick_params(**gconfig['tick_params'])
            plt.tight_layout()

        xticks += list(x) + [32]
        xtickspos += list(width + np.arange(len(x)+1))

    plt.subplots_adjust(**gconfig['subplots_adjust'])

    for file in gconfig['outfiles']:
        file = file.format(
            images_dir, this_filename[:-3], int(xticks[-1]), '-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        save_and_crop(fig, file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
