import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plot.plotting_utilities import *
import argparse

# python scripts/plot/approx_speedup.py -i results/final-v2/ -o temp -c lhc,sps,ps -s

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the intermediate effect analysis.',
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
        '0': '',
        '1': 'SRP',
        '2': 'RDS',
    },
    'label': {
        'double': 'Base',
        'single': 'F32',
        'singleSRP': 'F32-SRP',
        'doubleSRP': 'SRP',
        'singleRDS': 'F32-RDS',
        'doubleRDS': 'RDS',
    },
    'colors': {
        'Base': '0.8',
        'F32': '0.8',
        'SRP': '0.5',
        'F32-SRP': '0.5',
        'RDS': '0.3',
        'F32-RDS': '0.3',
    },
    'hatches': {
        'Base': '',
        'F32': '///',
        'SRP': '',
        'F32-SRP': '///',
        'RDS': '',
        'F32-RDS': '///',
    },

    # 'hatches': ['', '', '///', '', '///'],
    # 'colors': ['0.1', '0.5', '0.5', '0.8', '0.8'],
    'x_name': 'n',
    # 'x_to_keep': [16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    'xlabel': {
        'xlabel': 'Nodes (x20 Cores/ x1 GPU)',
        'fontsize': 10},
    'ylabel': {
        'ylabel': 'Norm. Runtime',
        'fontsize': 10,
        'labelpad': 3,
    },
    'title': {
        # 's': '',
        'fontsize': 10,
        'y': .85,
        # 'x': 0.1,
        'fontweight': 'bold',
    },
    'figsize': [5, 2.],
    'annotate': {
        'fontsize': 10,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'xticks': {'fontsize': 10, 'rotation': '0', 'fontweight': 'bold'},
    'ticks': {'fontsize': 10, 'rotation': '0'},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 9, 'handlelength': 1.5, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.3,
        'bbox_to_anchor': (-0.01, 1.15)
    },
    'subplots_adjust': {
        'wspace': 0.1, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 3, 'top': 0, 'bottom': 0, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    # 'ylim': [0.2, 1.0],
    # 'yticks': [0.2, 0.4, 0.6, 0.8, 1],
    'ylim': [0.25, 1.05],
    'yticks': [0.25, 0.5, 0.75, 1],
    'outfiles': [
        '{}/{}-{}-gpu.png',
        '{}/{}-{}-gpu.pdf'
    ],
    'files': [
        '{}/{}/exact-timing-gpu/comm-comp-report.csv',
        '{}/{}/rds-timing-gpu/comm-comp-report.csv',
        '{}/{}/srp-timing-gpu/comm-comp-report.csv',
        '{}/{}/float32-timing-gpu/comm-comp-report.csv',
        '{}/{}/f32-rds-timing-gpu/comm-comp-report.csv',
        '{}/{}/f32-srp-timing-gpu/comm-comp-report.csv',
    ],
    'lines': {
        # 'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
        # 'lb': ['reportonly'],
        'approx': ['0', '1', '2'],
        # 'red': ['1', '2', '3', '4'],
        'red': ['1', '3'],
        'prec': ['single', 'double'],
        'omp': ['20'],
        'N': ['8'],
        # 'ppb': ['4000000'],
        # 'lba': ['500'],
        # 'b': ['96', '48', '72', '21'],
        # 't': ['40000'],
        'type': ['total'],
    },
    # 'linefilter': [
    #     {'N': ['2', '8']},
    # ],

}

# Force sans-serif math mode (for axes labels)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}'
plt.rcParams['font.family'] = gconfig['fontname']


if __name__ == '__main__':

    # len + 1 for the avg
    fig, ax_arr = plt.subplots(ncols=len(args.cases)+1,
                           nrows=1,
                           sharex=False, sharey=True,
                           figsize=gconfig['figsize'])
    avg = {}
    step = 1.
    labels = set()

    for col, case in enumerate(args.cases+ ['AVG']):
        ax = ax_arr[col]
        plt.sca(ax)
        if col == 0:
            plt.ylabel(**gconfig['ylabel'])

        if case != 'AVG':

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
                                 linefilter=gconfig.get('linefilter', {}),
                                 prefix=True)
                for key in temp.keys():
                    if 'tp-approx' in file:
                        plots_dir['_{}_tp1'.format(key)] = temp[key].copy()
                    else:
                        plots_dir['_{}_tp0'.format(key)] = temp[key].copy()


            # First the reference value
            keyref = ''
            for k in plots_dir.keys():
                if 'approx0' in k and 'double' in k:
                    keyref = k
                    break
            if keyref == '':
                print('ERROR: reference key not found')
                exit(-1)

            refvals = plots_dir[keyref]
            # del plots_dir[keyref]


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
            print(plots_dir.keys())
            keys = ['_'.join(a.split('_')[1:4]) for a in list(plots_dir.keys())]
            print(keys)
            keys = np.array(list(plots_dir.keys()))[np.argsort(keys)]
            
            width = step / (len(plots_dir.keys()) + 1)
            
            for idx, k in enumerate(keys):
                values = plots_dir[k]
                approx = k.split('approx')[1].split('_')[0]
                red = k.split('red')[1].split('_')[0]
                experiment = k.split('_')[-1]
                prec = k.split('prec')[1].split('_')[0]
                approx = gconfig['approx'][approx]
                label = gconfig['label'][prec+approx]

                # if approx == 'SRP':
                #     label += '-{}'.format(red)
                # if label == 'base':
                #     continue

                x = get_values(values, header, gconfig['x_name'])
                omp = get_values(values, header, gconfig['omp_name'])
                y = get_values(values, header, gconfig['y_name'])
                parts = get_values(values, header, 'ppb')
                bunches = get_values(values, header, 'b')
                turns = get_values(values, header, 't')

                # This is the throughput
                y = parts * bunches * turns / y

                speedup = []
                j = 0
                for i, xiref in enumerate(xref):
                    if j < len(x) and xiref == x[j]:
                        speedup.append(yref[i]/y[j])
                        j += 1
                    else:
                        speedup.append(0)
                speedup = np.array(speedup)

                x = xref * omp[0]

                if label not in avg:
                    avg[label] = []
                avg[label].append(speedup)

                legend = label
                if label in labels:
                    legend = None
                else:
                    labels.add(label)

                plt.bar(idx*width + np.arange(len(speedup)), 
                        speedup, width=0.74*width,
                        edgecolor='0.', label=legend, 
                        hatch=gconfig['hatches'][label],
                        color=gconfig['colors'][label])


        elif case == 'AVG':
            width = step / (len(avg.keys()) + 1)
            for idx, key in enumerate(avg.keys()):
                vals = avg[key]
                val = np.mean(vals)
                plt.bar(idx*width, val, width=.75 * width,
                        edgecolor='0.', label=None,
                        hatch=gconfig['hatches'][key],
                        color=gconfig['colors'][key])
                text = '{:.2f}'.format(val)
                if idx == 0:
                    text = ''
                else:
                    text = text[:]
                ax.annotate(text, xy=(idx*width, 0.01 + val),
                            rotation='90',
                            **gconfig['annotate'])
        else:
            print('Error, case {} not found'.format(case))
            exit(-1)

        plt.grid(True, which='major', alpha=0.5)
        plt.grid(False, which='major', axis='x')
        plt.gca().set_axisbelow(True)

        # plt.title('{}'.format(case.upper()), **gconfig['title'])
        plt.xticks([idx*width/2], [case.upper()], **gconfig['xticks'])

        if col == 0:
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles=handles, labels=labels, **gconfig['legend'])

        # plt.xlim(xmin=0-width, xmax=len(xref) - width)
        plt.ylim(*gconfig['ylim'])
        plt.yticks(gconfig['yticks'], **gconfig['ticks'])

        # if col == 1:
        #     plt.xlabel(**gconfig['xlabel'])

        ax.tick_params(**gconfig['tick_params'])
        plt.tight_layout()

    plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(
            images_dir, this_filename[:-3], '-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        save_and_crop(fig, file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
