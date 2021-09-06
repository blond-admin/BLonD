import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse

# python scripts/plot/cpu_vs_gpu_weak_scaling.py -i results/final-v2/ -o results/final-v2/plots/ -s

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
        'exact-gpu0-tp0': 'HBLonD',
        # 'exact-gpu0-tp1': 'CPU-TP',
        'exact-gpu1-tp0': 'CuBLonD-1PN',
        'exact-gpu2-tp0': 'CuBLonD-2PN',

    },

    'colors': {
        'HBLonD': '0.9',
        'CuBLonD-1PN': '0.5',
        'CuBLonD-2PN': '0.2',

    },
    'hatches': {
        'HBLonD': '',
        'CuBLonD-1PN': '',
        'CuBLonD-2PN': '',

    },

    'x_name': 'n',
    # 'x_to_keep': [4, 8, 16, 32, 64],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    'xlabel': '1 Full Node (x20 Cores/ x1 or x2 GPUs)',
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
        'fontsize': 10,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center',
        'rotation': '0',
    },
    'ticks': {'fontsize': 10},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 4, 'handlelength': 1.6, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        'bbox_to_anchor': (0, 1.21),
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params_left': {
        'pad': 3, 'top': 0, 'bottom': 0, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'tick_params_center_right': {
        'pad': 3, 'top': 0, 'bottom': 0, 'left': 0,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',

    'ylim': [0., 10],
    'yticks': [0, 2, 4, 6, 8, 10],
    # 'yticks2': [0, 20, 40, 60, 80, 100],
    'outfiles': ['{}/{}-{}.png', '{}/{}-{}.pdf'],
    'files': [
        # '{}/{}/tp-approx0-weak-scaling/comm-comp-report.csv',
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
        'N': ['1'],
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
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}'
plt.rcParams['font.family'] = gconfig['fontname']

if __name__ == '__main__':

    avg = {}
    fig, ax = plt.subplots(ncols=1, nrows=1,
                           sharex=True, sharey=True,
                           figsize=gconfig['figsize'])
    
    labels = set()
    plt.sca(ax)
    step = 1
    pos = 0
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


        keyref = ''
        for k in plots_dir.keys():
            if k == 'HBLonD':
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

        width = .8 / len(plots_dir.keys())
        
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))
        xticks.append(case.upper())
        xtickspos.append(pos+width)
        for idx, k in enumerate(gconfig['label'].values()):
            if k not in plots_dir:
                continue
            values = plots_dir[k]
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
            x = x * omp[0]
            speedup = speedup / yref
            # speedup = speedup / speedup[0]
            if label not in avg:
                avg[label] = []
            avg[label].append(speedup)

            if label in labels:
                legend = None
            else:
                legend = label
                labels.add(label)

            plt.bar(pos + idx * width + np.arange(len(x)), speedup, width=0.85*width,
                    edgecolor='0', label=legend, hatch=gconfig['hatches'][k],
                    color=gconfig['colors'][k])
            
            # if 'CPU-BASE':
            for i in np.arange(len(speedup)):
                # if speedup[i] > 0.9:
                #     continue
                ax.annotate('{:.1f}'.format(speedup[i]),
                            xy=(pos+idx * width + i, speedup[i]),
                            **gconfig['annotate'])
            # pos += 1 * width

        plt.axvline(x=(pos+idx*width + pos+step)/2, color='black', ls='--')

        pos += step

    xticks.append('AVG')
    xtickspos.append(pos+width)
    for idx, key in enumerate(avg.keys()):
        vals = avg[key]
        val = np.mean(vals)
        plt.bar(pos + idx*width, val, width=.85 * width,
                edgecolor='0.', label=None,
                hatch=gconfig['hatches'][key],
                color=gconfig['colors'][key])
        text = '{:.1f}'.format(val)
        # if idx == 0:
        #     text = ''
        # else:
        #     text = text[:]
        ax.annotate(text, xy=(pos + idx*width, 0.01 + val),
                    **gconfig['annotate'])
    pos += step
    plt.grid(True, which='major', alpha=0.5)
    plt.grid(False, which='major', axis='x')
    plt.gca().set_axisbelow(True)

    # plt.title('{}'.format(case.upper()), **gconfig['title'])
    # if col == 1:
    plt.xlabel(gconfig['xlabel'], labelpad=3,
               fontweight='bold',
               fontsize=gconfig['fontsize'])
    # if col == 0:
    plt.ylabel(gconfig['ylabel'], labelpad=3,
               fontweight='bold',
               fontsize=gconfig['fontsize'])
    plt.xticks((pos-width)/2 + np.arange(len(x)), np.array(x, int)//20, **gconfig['ticks'])
    ax.tick_params(**gconfig['tick_params_left'])
    plt.legend(**gconfig['legend'])

    plt.ylim(gconfig['ylim'])
    plt.yticks(gconfig['yticks'], **gconfig['ticks'])
    plt.xticks(xtickspos, xticks, **gconfig['ticks'])
    # plt.xlim(0-width/2, pos-)
    plt.tight_layout()
    # plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(images_dir, this_filename[:-3], '-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        save_and_crop(fig, file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
