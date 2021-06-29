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
        'Base': 'tab:orange',
        'F32': 'tab:orange',
        'F32-SRP': 'tab:blue',
        'SRP': 'tab:blue',
        'F32-RDS': 'tab:green',
        'RDS': 'tab:green',
    },
    'markers': {
        'Base': '',
        'F32': 'x',
        'F32-SRP': 'x',
        'SRP': '',
        'F32-RDS': 'x',
        'RDS': '',
    },


    # 'hatches': ['', '', 'xx', '', 'xx', '', 'xx', '', 'xx'],
    # 'colors': ['0.1', '0.3', '0.3', '0.5', '0.5', '0.7', '0.7', '0.95', '0.95'],
    'x_name': 'n',
    # 'x_to_keep': [16],
    'omp_name': 'omp',
    'y_name': 'avg_time(sec)',
    'xlabel': {
        'xlabel': 'Nodes (x20 Cores)'
    },
    'ylabel': 'Norm. Throughput',
    'title': {
        # 's': '',
        'fontsize': 10,
        'y': .96,
        'x': 0.1,
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
        'loc': 'upper left', 'ncol': 9, 'handlelength': 1.6, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.3,
        'bbox_to_anchor': (-0.01, 1.17)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 0, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    # 'ylim': [0.5, 48],
    'ylim': [0., 1.1],
    'yticks': [0., 0.2, 0.4, 0.6, 0.8, 1.0],

    # 'ylim': [0, 35],
    # 'yticks': [0, 5, 10, 15, 20, 25, 30, 35],
    'outfiles': [
        '{}/{}-{}.png',
        '{}/{}-{}.pdf'
    ],
    'files': [
        '{}/{}/exact-timing-cpu/comm-comp-report.csv',
        '{}/{}/rds-timing-cpu/comm-comp-report.csv',
        '{}/{}/srp-timing-cpu/comm-comp-report.csv',
        '{}/{}/float32-timing-cpu/comm-comp-report.csv',
        '{}/{}/f32-rds-timing-cpu/comm-comp-report.csv',
        '{}/{}/f32-srp-timing-cpu/comm-comp-report.csv',

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
        'omp': ['10'],
        # 'N': ['8'],
        # 'ppb': ['4000000'],
        # 'lba': ['500'],
        # 'b': ['96', '48', '72', '21'],
        # 't': ['40000'],
        'type': ['total'],
    },
    # 'reference': {
    #     'file': '{}/{}/cpu-baseline/comm-comp-report.csv',
    #     'lines': {
    #         'b': ['12', '21', '18'],
    #         'ppb': ['1000000', '1500000', '16000000'],
    #         # 't': ['5000'],
    #         'type': ['total'],
    #     },
    # }

}

# Force sans-serif math mode (for axes labels)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}'
plt.rcParams['font.family'] = gconfig['fontname']


if __name__ == '__main__':

    fig, ax = plt.subplots(ncols=1, nrows=1,
                           sharex=True, sharey=True,
                           figsize=gconfig['figsize'])
    plt.sca(ax)

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

        width = .85 * step / (len(plots_dir.keys()))

        # data = np.genfromtxt(gconfig['reference']['file'].format(res_dir, case),
        #                      delimiter='\t', dtype=str)
        # header, data = list(data[0]), data[1:]
        # ref_dir = get_plots(header, data, gconfig['reference']['lines'],
        #                     exclude=gconfig.get('exclude', []),
        #                     prefix=True)

        # print('Ref keys: ', ref_dir.keys())
        # ref_dir_keys = list(ref_dir.keys())
        # # assert len(ref_dir_keys) == 1
        # refvals = ref_dir[ref_dir_keys[-1]]

        # xref = get_values(refvals, header, gconfig['x_name'])
        # omp = get_values(refvals, header, gconfig['omp_name'])
        # y = get_values(refvals, header, gconfig['y_name'])
        # parts = get_values(refvals, header, 'ppb')
        # bunches = get_values(refvals, header, 'b')
        # turns = get_values(refvals, header, 't')
        # yref = parts * bunches * turns / y
        # yref /= (x * omp // 20)


        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))
        # To sort the keys, by approx and then reduce value
        print(plots_dir.keys())
        keys = ['_'.join(a.split('_')[1:4]) for a in list(plots_dir.keys())]
        print(keys)
        keys = np.array(list(plots_dir.keys()))[np.argsort(keys)]
        for idx, k in enumerate(keys):
            values = plots_dir[k]
            # mpiv = k.split('_mpi')[1].split('_')[0]
            # lb = k.split('lb')[1].split('_')[0]
            approx = k.split('approx')[1].split('_')[0]
            # tp = k.split('_')[-1]
            red = k.split('red')[1].split('_')[0]
            experiment = k.split('_')[-1]
            prec = k.split('prec')[1].split('_')[0]
            # if lb == 'interval':
            #     lb = 'LB-'
            # elif lb == 'reportonly':
            #     lb = ''
            # if tp == 'tp1':
            #     tp = 'TP-'
            # elif tp == 'tp0':
            #     tp = ''
            approx = gconfig['approx'][approx]
            label = gconfig['label'][prec+approx]
            # if prec == 'single':
            #     label = 'f32'
            # if approx == '':
            #     label = 'base'
            # elif approx == 'RDS':
            #     label += 'RDS'
            # elif approx == 'SRP':
            #     label += 'SRP-{}'.format(red)
            # if approx == 'SRP':
            # label += '-{}'.format(red)
            # if label == 'base':
            # continue

            # if label == '1':
            #     label = 'base'
            # if label[-1] == '-':
            #     label = label[:-1]

            x = get_values(values, header, gconfig['x_name'])
            omp = get_values(values, header, gconfig['omp_name'])
            y = get_values(values, header, gconfig['y_name'])
            parts = get_values(values, header, 'ppb')
            bunches = get_values(values, header, 'b')
            turns = get_values(values, header, 't')

            # This is the throughput
            y = parts * bunches * turns / y
            y /= (x * omp //20)
            # y = y[1:]
            speedup = y / y[0]
            x = x * omp // 20
            # x = x[1:]
            # speedup = []
            # j = 0
            # for i, xiref in enumerate(xref):
            #     if j < len(x) and xiref == x[j]:
            #         speedup.append(y[j]/yref[i])
            #         j += 1
            #     else:
            #         speedup.append(0)
            # speedup = np.array(speedup)
            # x = xref * omp[0]

            # if label not in avg:
            #     avg[label] = []
            # avg[label].append(speedup)
            # efficiency = 100 * speedup / x
            legend = label
            if label in labels:
                legend = None
            else:
                labels.add(label)

            # x = x[1:]
            # speedup = speedup[1:]
            plt.errorbar(pos+np.arange(len(x)), speedup,
                         yerr=None,
                         label=legend,
                         lw=1.2,
                         color=gconfig['colors'][label],
                         marker=gconfig['markers'][label],
                         capsize=2)
            # if k != keyref:
            #     for i in np.arange(len(speedup)):
            #         if speedup[i] > 0.9:
            #             continue
            #         ax.annotate('{:.2f}'.format(speedup[i]),
            #                     xy=(pos+idx*width+i, speedup[i]),
            #                     rotation='90', **gconfig['annotate'])
        xticks += list(x)
        xtickspos += list(pos + np.arange(len(x)))
        if case != args.cases[-1]:
            plt.axvline(x=pos + len(x)-step/2, color='black', ls='--')
        ax.annotate('{}'.format(case.upper()),
                    xy=(pos + (len(x)-1)/2, gconfig['ylim'][0]+0.3),
                    **gconfig['annotate'])
        pos += len(x)

    # for idx, key in enumerate(avg.keys()):
    #     vals = avg[key]
    #     val = np.mean(vals)
    #     plt.bar(pos + idx*width, val, width=.75 * width,
    #             edgecolor='0.', label=None,
    #             hatch=gconfig['hatches'][key],
    #             color=gconfig['colors'][key])
    #     text = '{:.2f}'.format(val)
    #     if idx == 0:
    #         text = ''
    #     else:
    #         text = text[:]
    #     ax.annotate(text, xy=(pos + idx*width, 0.01 + val),
    #                 rotation='90',
    #                 **gconfig['annotate'])
    # pos += step

    # plt.yscale('log', base=2)

    handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    plt.grid(True, which='major', alpha=0.5)
    plt.grid(False, which='major', axis='x')
    plt.gca().set_axisbelow(True)

    plt.ylabel(gconfig['ylabel'], labelpad=3)
               # fontweight='bold',
               # fontsize=gconfig['fontsize'])

    # plt.title('{}'.format(case.upper()), **gconfig['title'])

    plt.legend(handles=handles, labels=labels, **gconfig['legend'])

    plt.ylim(gconfig['ylim'])
    plt.yticks(gconfig['yticks'], **gconfig['ticks'])
    plt.xticks(xtickspos, np.array(xticks, int), **gconfig['xticks'])
    # plt.xticks(np.arange(pos) + step/3,
    #            [c.upper() for c in args.cases] + ['AVG'], **gconfig['xticks'])
    # plt.xlim(0-1.3*width/2, pos-2.6*width/2)
    plt.xlabel(**gconfig['xlabel'])

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
