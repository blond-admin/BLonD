import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
import sys
from plot.plotting_utilities import *
import argparse


this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Generate the figure of the run time breakdown.',
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
        '0': '',
        '1': 'SRP',
        '2': 'RDS',
    },
    'label': {
        'double': 'base',
        'single': 'f32',
        'singleSRP': 'f32-SRP',
        'doubleSRP': 'SRP',
        'singleRDS': 'f32-RDS',
        'doubleRDS': 'RDS',
    },
    'markers': ['x', 'o', '^'],
    'edgecolors': ['xkcd:red', 'xkcd:blue'],
    'f_name': 'function',
    'x_name': 'n',
    # 'x_to_keep': [4, 8, 16, 32, 64],
    # 'omp_name': 'omp',
    'y_name': 'total_time(sec)',
    'xlabel': 'Workers (x1 GPU/ x20 Cores)',
    'ylabel': r'Norm. Runtime',
    'title': {
        # 's': '{}'.format(),
        'fontsize': 10,
        # 'y': .82,
        # 'x': 0.55,
        'fontweight': 'bold',
    },
    'figsize': [6.4, 2.2],
    'annotate': {
        'fontsize': 8.5,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 10},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 10, 'handlelength': 1.5, 'fancybox': False,
        'framealpha': 0.8,'frameon': False, 'fontsize': 9, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.5, 'borderaxespad': 0.1, 'columnspacing': 0.8,
        'bbox_to_anchor': (0, 1.25)
    },
    'subplots_adjust': {
        'wspace': 0.1, 'hspace': 0.1,
        # 'top': 0.93
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
    'phases': ['comm', 'serial'],
    'ylim': [0, 1.4],
    'xlim': [1.6, 36],
    'yticks': [0, 20, 40, 60, 80, 100],
    'outfiles': [
        '{}/{}-{}-{}-gpu.png',
        # '{}/{}-{}-{}-gpu.pdf'
    ],
    'files': [
        '{}/{}/exact-timing-gpu/avg-report.csv',
        # '{}/{}/rds-timing-gpu/avg-report.csv',
        # '{}/{}/srp-timing-gpu/avg-report.csv',
        '{}/{}/float32-timing-gpu/avg-report.csv',
        # '{}/{}/f32-rds-timing-gpu/avg-report.csv',
        # '{}/{}/lb-tp-approx1-strong-scaling/comm-comp-report.csv',
    ],
    'lines': {
        # 'mpi': ['mpich3', 'mvapich2', 'openmpi3'],
        # 'lb': ['reportonly'],
        'approx': ['0', '1', '2'],
        # 'red': ['1', '2', '3', '4'],
        'red': ['1', '2', '3'],
        'prec': ['single', 'double'],
        'omp': ['20'],
        # 'ppb': ['4000000'],
        # 'lba': ['500'],
        # 'b': ['96', '48', '72', '21'],
        # 't': ['40000'],
        # 'type': ['total'],
    },
    'categories': {
        'LIKick': ['comp:LIKick'],
        'kick': ['comp:kick'],
        'drift': ['comp:drift'],
        # 'histo': ['comp:histo', 'serial:scale_histo'],
        'iv1turn': ['serial:indVolt1Turn'],
        'bspectr': ['serial:beam_spectrum_gen'],
        # 'indVolt': ['serial:InductiveImped', 'serial:beam_spectrum_gen',
        #             'serial:indVolt1Turn', 'serial:shift_trev_time'],
        'updateImp': ['serial:updateImp'],
        'total': ['total_time'],
        'other': ['serial:beam_phase', 'serial:phase_difference', 'serial:RFVCalc', 'serial:binShift']
    },


    'ch': {
        'total': ['0.1', ''],
        'kick': ['0.1', 'xx'],
        'drift': ['0.3', ''],
        # 'histo': ['0.3', 'xx'],
        # 'indVolt': ['0.5', ''],
        'iv1turn': ['0.3', 'xx'],
        'bspectr': ['0.5', ''],
        'updateImp': ['0.5', 'xx'],
        'other': ['0.95', ''],
        'LIKick': ['0.9', 'xx'],
    },


}

# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
# # Force sans-serif math mode (for axes labels)
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
# plt.rcParams['font.family'] = 'sans-serif'  # ... for regular text
# plt.rcParams['font.sans-serif'] = 'Helvetica'


if __name__ == '__main__':
    for col, case in enumerate(args.cases):
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Reading data'))

        # ax = ax_arr[col]
        # plt.sca(ax)
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
                plots_dir['_{}'.format(key)] = temp[key].copy()


            final_dir = {}
            for idx, k in enumerate(plots_dir.keys()):
                approx = k.split('approx')[1].split('_')[0]
                red = k.split('red')[1].split('_')[0]
                experiment = k.split('_')[-1]
                prec = k.split('prec')[1].split('_')[0]
                approx = gconfig['approx'][approx]
                name = gconfig['label'][prec+approx]
                if approx == 'SRP':
                    name += '-{}'.format(red)

                if name not in final_dir:
                    final_dir[name] = {}
                for row in plots_dir[k]:
                    func = row[header.index(gconfig['f_name'])]
                    workers = row[header.index(gconfig['x_name'])]
                    if workers not in final_dir[name]:
                        final_dir[name][workers] = {}
                    for cat, subcat in gconfig['categories'].items():
                        if func in subcat:
                            if cat not in final_dir[name][workers]:
                                final_dir[name][workers][cat] = 0
                            final_dir[name][workers][cat] += float(
                                row[header.index(gconfig['y_name'])])

        for name in final_dir.keys():
            if name == 'base':
                continue
            
            fig, ax = plt.subplots(ncols=1, nrows=1,
                                   sharex=True, sharey=True,
                                   figsize=gconfig['figsize'])
            # step = 1
            pos = 0
            width = 1.
            # width = step/3
            xticks = [[], []]
            labels = set()
            offset = 1. / (len(final_dir[name])+1.)
            for workers, funcs in final_dir[name].items():
                xticks[0].append(pos)
                xticks[1].append(workers)
                for func, val in funcs.items():
                    if func not in labels:
                        labels.add(func)
                        label = func
                    else:
                        label = None
                    normval = val / final_dir['base'][workers][func]
                    plt.bar(pos, normval, width=.9*width,
                            edgecolor='0', color=gconfig['ch'][func][0],
                            label=label, hatch=gconfig['ch'][func][1])
                    ax.annotate('{:.2f}'.format(normval), xy=(pos, normval),
                                **gconfig['annotate'])

                    pos += width
                pos += 0.5*width

            # for cat in gconfig['categories'].keys():
            #     # offset = 0
            #     for prec in ['f64', 'f32']:
            #         # for k, v in final_dir[]
            #         if prec not in labels:
            #             label = prec
            #             labels.add(label)
            #         else:
            #             label = None
            #         val = final_dir[cat] / final_dir[cat]
            #         plt.bar(pos+offset, val, width=.9*width,
            #                 edgecolor='0', color=gconfig['colors'][prec],
            #                 label=label)
            #         ax.annotate('{:.2f}'.format(val), xy=(pos+offset, val),
            #                     **gconfig['annotate'])
            #         offset += width
            #     xticks.append(cat)
            #     pos += step

                # plt.pie(final_dir.values(), labels=final_dir.keys(),
                #         explode=[0.01]*len(final_dir),
                #         autopct='%1.1f', pctdistance=0.8,
                #         labeldistance=1.1)
                # ax.axis('equal')
            # plt.title(f'{case.upper()}-{name}', **gconfig['title'])
            plt.scatter([],[], s=0, label=f'{case.upper()}-{name}')
            # plt.ylim(gconfig['ylim'])
            plt.ylabel(gconfig['ylabel'], labelpad=2, color='xkcd:black',
                       fontsize=gconfig['fontsize'])
            plt.xlabel(gconfig['xlabel'], labelpad=2, color='xkcd:black',
                       fontsize=gconfig['fontsize'])

            plt.xticks(np.array(xticks[0])+(xticks[0][1]-xticks[0][0])/2, xticks[1], **gconfig['ticks'])
            plt.legend(**gconfig['legend'])

            plt.tight_layout()

            # plt.subplots_adjust(**gconfig['subplots_adjust'])
            for file in gconfig['outfiles']:
                file = file.format(
                    images_dir, this_filename[:-3], name, case)
                print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
                save_and_crop(fig, file, dpi=600, bbox_inches='tight')
            if args.show:
                plt.show()
            plt.close()
