import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse


# python scripts/plot/approx_strong_scaling_gpu.py -icpu results/weak-scaling-cpu/ -igpu results/final-v2/ -b results/baselinecpu/ -o results/multiplatform/plots/ -s

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the intermediate effect analysis.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-icpu', '--inputcpu', type=str, default=os.path.join(project_dir, 'results'),
                    help='The directory with the CPU results.')

parser.add_argument('-igpu', '--inputgpu', type=str, default=os.path.join(project_dir, 'results'),
                    help='The directory with the GPU results.')


parser.add_argument('-b', '--basedir', type=str, default=os.path.join(project_dir, 'results'),
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
    images_dir = os.path.join(res_dir, 'plots')
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
        'doublecpu': 'CPU-Base',
        'singlecpu': 'CPU-F32',
        'singleSRPcpu': 'CPU-F32-SRP',
        'doubleSRPcpu': 'CPU-SRP',
        'singleRDScpu': 'CPU-F32-RDS',
        'doubleRDScpu': 'CPU-RDS',
        'doublegpu': 'GPU-Base',
        'singlegpu': 'GPU-F32',
        'singleSRPgpu': 'GPU-F32-SRP',
        'doubleSRPgpu': 'GPU-SRP',
        'singleRDSgpu': 'GPU-F32-RDS',
        'doubleRDSgpu': 'GPU-RDS',


    },
    'colors': {
        'CPU-Base': '0.5',
        'CPU-F32-RDS': '0.5',
        'CPU-F32-SRP': '0.5',

        'GPU-Base': '0.',
        'GPU-F32-SRP': '0.',
        'GPU-F32-RDS': '0.',
        

        # 'CPU-F32': 'tab:orange',
        # 'CPU-SRP': 'tab:blue',
        # 'CPU-RDS': 'tab:green',
    },
    'markers': {
        'CPU-Base': '',
        'CPU-F32-SRP': '.',
        'CPU-F32-RDS': '*',
        
        'GPU-Base': '',
        'GPU-F32-SRP': '.',
        'GPU-F32-RDS': '*',


        # 'CPU-SRP': '',
        # 'CPU-F32': 'x',
        # 'CPU-RDS': '',
    },


    # 'hatches': ['', '', 'xx', '', 'xx', '', 'xx', '', 'xx'],
    # 'colors': ['0.1', '0.3', '0.3', '0.5', '0.5', '0.7', '0.7', '0.95', '0.95'],
    'x_name': 'N',
    # 'x_to_keep': [16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    'xlabel': {
        'xlabel': 'Nodes (x20 Cores/ x1 GPU)'
    },
    'ylabel': 'Speedup',
    'title': {
        # 's': '',
        'fontsize': 10,
        'y': 0.85,
        'x': 0.5,
        'fontweight': 'bold',
    },
    'figsize': [5, 2.],
    'annotate': {
        'fontsize': 10,
        'textcoords': 'data',
        'va': 'top',
        'ha': 'center'
    },
    'xticks': {'fontsize': 10, 'rotation': '0'},
    'ticks': {'fontsize': 10, 'rotation': '0'},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 3, 'handlelength': 2, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.4, 'borderaxespad': 0.1, 'columnspacing': 0.5,
        'bbox_to_anchor': (-0.01, 1.21)
    },
    'subplots_adjust': {
        'wspace': 0.1, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 2, 'top': 0, 'bottom': 1, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    # 'ylim': [0.5, 35],
    'ylim': [0, 45],
    'yticks': [0, 10,  20, 30,  40],
    # 'yticks': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    'outfiles': [
        '{}/{}-{}-nodes-{}.png',
        '{}/{}-{}-nodes-{}.pdf'
    ],
    'files': [
        '{}/{}/exact-timing-{}/comm-comp-report.csv',
        # '{}/{}/rds-timing-gpu/comm-comp-report.csv',
        # '{}/{}/srp-timing-gpu/comm-comp-report.csv',
        # '{}/{}/float32-timing-gpu/comm-comp-report.csv',
        '{}/{}/f32-rds-timing-{}/comm-comp-report.csv',
        '{}/{}/f32-srp-timing-{}/comm-comp-report.csv',

        # '{}/{}/exact-timing-gpu/comm-comp-report.csv',
        # '{}/{}/rds-timing-gpu/comm-comp-report.csv',
        # '{}/{}/srp-timing-gpu/comm-comp-report.csv',
        # '{}/{}/float32-timing-gpu/comm-comp-report.csv',
        # '{}/{}/f32-rds-timing-gpu/comm-comp-report.csv',
        # '{}/{}/f32-srp-timing-gpu/comm-comp-report.csv',
    ],
    'lines': {
        # 'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
        # 'lb': ['reportonly'],
        'approx': ['0', '1', '2'],
        # 'red': ['1', '2', '3', '4'],
        'red': ['1', '3'],
        'prec': ['single', 'double'],
        'omp': ['10', '20'],
        'gpu': ['0', '1'],
        # 'N': ['1', '2', '4', '8', '16', '32'],
        # 'ppb': ['4000000'],
        # 'lba': ['500'],
        # 'b': ['96', '48', '72', '21'],
        # 't': ['40000'],
        'type': ['total'],
    },
    'linefilter': [
        {'N': ['20', '32']},
    ],
    'reference': {
        'file': '{}/{}/comm-comp-report.csv',
        'lines': {
            # 'b': ['12', '21', '18'],
            # 'b': ['192', '288', '21'],
            'b': ['24', '36', '21'],
            'ppb': ['1500000', '1000000', '2000000'],
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

    fig, ax_arr = plt.subplots(ncols=len(args.cases), nrows=1,
                           sharex=False, sharey=True,
                           figsize=gconfig['figsize'])
    # pos = 0
    step = 1.
    labels = set()
    xticks = []
    xtickspos = []
    for col, case in enumerate(args.cases):
        ax = ax_arr[col]
        plt.sca(ax)

        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Reading data'))
        plots_dir = {}
        for indir, platform in zip([args.inputcpu, args.inputgpu], ['cpu', 'gpu']):
            plots_dir[platform] = {}
            for file in gconfig['files']:
                file = file.format(indir, case, platform)
                # print(file)
                data = np.genfromtxt(file, delimiter='\t', dtype=str)
                header, data = list(data[0]), data[1:]
                temp = get_plots(header, data, gconfig['lines'],
                                 exclude=gconfig.get('exclude', []),
                                 linefilter=gconfig.get('linefilter', {}),
                                 prefix=True)
                for key in temp.keys():
                    if 'tp-approx' in file:
                        plots_dir[platform]['_{}_tp1'.format(key)] = temp[key].copy()
                    else:
                        plots_dir[platform]['_{}_tp0'.format(key)] = temp[key].copy()

        data = np.genfromtxt(gconfig['reference']['file'].format(args.basedir, case),
                             delimiter='\t', dtype=str)
        header, data = list(data[0]), data[1:]
        ref_dir = get_plots(header, data, gconfig['reference']['lines'],
                            exclude=gconfig.get('exclude', []),
                            linefilter=gconfig.get('linefilter', []),
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
        for platform in plots_dir.keys():
            for idx, k in enumerate(plots_dir[platform].keys()):
                values = plots_dir[platform][k]
                approx = k.split('approx')[1].split('_')[0]
                red = k.split('red')[1].split('_')[0]
                experiment = k.split('_')[-1]
                prec = k.split('prec')[1].split('_')[0]
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
                print("{}:{}:{}:{:.2f}".format(platform, case, label, speedup[-1]))
                plt.errorbar(np.arange(len(x)), speedup,
                             yerr=None,
                             label=legend,
                             lw=1.2,
                             color=gconfig['colors'][label],
                             marker=gconfig['markers'][label],
                             capsize=2)
        xticks += list(x)
        xtickspos += list(np.arange(len(x)))
        # if case != args.cases[-1]:
        #     plt.axvline(x=pos + len(x)-step/2, color='black', ls='--')
        # ax.annotate('{}'.format(case.upper()),
        #             xy=(pos + (len(x)-1)/2, gconfig['ylim'][-1]-2),
        #             **gconfig['annotate'])
        # pos += len(x)


        # print(labels)
        plt.grid(True, which='major', alpha=0.5)
        plt.grid(False, which='major', axis='x')
        plt.gca().set_axisbelow(True)

        if col == 0:
            handles, labs = ax.get_legend_handles_labels()
            for i in range(len(labs)):
                print('{}:{}'.format(i, labs[i]))
            handles = [handles[i//2] if (i % 2 == 0) else handles[i+(len(handles)-i)//2] for i in range(len(handles))]
            labs = [labs[i//2] if (i % 2 == 0) else labs[i+(len(labs)-i)//2] for i in range(len(labs))]
            # handles = [handles[2*i] if i < 3 else handles[i] for i in range(len(handles))]
            # labs = [labs[2*i] if i < 3 else labs[i] for i in range(len(labs))]

            plt.ylabel(gconfig['ylabel'], labelpad=3)
            plt.legend(handles = handles, labels=labs, **gconfig['legend'])

        plt.title('{}'.format(case.upper()), **gconfig['title'])

        plt.ylim(gconfig['ylim'])
        plt.yticks(gconfig['yticks'], **gconfig['ticks'])
        plt.xticks(xtickspos, np.array(xticks, int), **gconfig['xticks'])
        # plt.xlim(xtickspos[0]-0.5, xtickspos[-1]+0.5)
        if col == 1:
            plt.xlabel(**gconfig['xlabel'])

        ax.tick_params(**gconfig['tick_params'])
        plt.tight_layout()
    plt.subplots_adjust(**gconfig['subplots_adjust'])

    for file in gconfig['outfiles']:
        file = file.format(
            images_dir, this_filename[:-3], int(xticks[-1]),'-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        save_and_crop(fig, file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
