import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
import sys
from plot.plotting_utilities import *
import argparse
# python scripts/plot/time_breakdown_approx_gpu.py -i results/aris-time-breakdown/ -s -o results/aris-time-breakdown/plots/

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
        'double': 'Base',
        'singleSRP': 'F32-SRP',
        'singleRDS': 'F32-RDS',
        # 'single': 'F32',
        # 'doubleSRP': 'SRP',
        # 'doubleRDS': 'RDS',
    },
    'colors': {
        'Base': '0.85',
        'F32': '0.85',
        'F32-SRP': '0.55',
        'SRP': '0.55',
        'F32-RDS': '0.3',
        'RDS': '0.3',
    },
    'alpha': {
        'Base': 1,
        'F32': 0.5,
        'F32-SRP': 1,
        'SRP': 0.5,
        'F32-RDS': 1,
        'RDS': 0.5,
    },
    # 'edgecolors': {
    #     'Base': 'tab:orange',
    #     'F32': 'tab:orange',
    #     'F32-SRP': 'tab:blue',
    #     'SRP': 'tab:blue',
    #     'F32-RDS': 'tab:green',
    #     'RDS': 'tab:green',
    # },
    # 'hatches': {
    #     'Base': '',
    #     'F32': 'xx',
    #     'F32-SRP': 'xx',
    #     'SRP': '',
    #     'F32-RDS': 'xx',
    #     'RDS': '',
    # },
    'hatches': {
        'comm': '...',
        'serial': '///',
        'comp': 'x'
    },

    'x_name': 'n',
    # 'x_to_keep': [4, 8, 16, 32, 64],
    'omp_name': 'omp',
    'y_name': 'percent',
    'xlabel': r'Nodes (x1 K40 GPU per Node)',
    'ylabel': r'Runtime (\%)',
    'title': {
                # 's': '{}'.format(case.upper()),
                'fontsize': 10,
                'y': .82,
                # 'x': 0.55,
                'fontweight': 'bold',
    },
    'figsize': [5, 1.8],
    'annotate': {
        'fontsize': 10,
        'textcoords': 'data',
        'rotation': 90,
        'va': 'bottom',
        'ha': 'center'
    },
    'ticks': {'fontsize': 10},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 10, 'handlelength': 1.3, 'fancybox': False,
        'framealpha': 0., 'fontsize': 9.5, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.4,
        'bbox_to_anchor': (-0.05, 1.14)
    },
    'subplots_adjust': {
        'wspace': 0.05, 'hspace': 0.1, 'top': 0.93
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
    'phases': ['comm', 'serial', 'comp'],
    'ylim': [0, 105],
    'xlim': [1.6, 36],
    'yticks': [0, 20, 40, 60, 80, 100],
    'outfiles': ['{}/{}-{}.png',
                 '{}/{}-{}.pdf'],
    'files': [
        '{}/{}/exact-timing-strong-gpu/comm-comp-report.csv',
        '{}/{}/f32-srp-timing-strong-gpu/comm-comp-report.csv',
        '{}/{}/f32-rds-timing-strong-gpu/comm-comp-report.csv',
    ],
    'lines': {
        'approx': ['0', '1', '2'],
        'red': ['1', '3'],
        'prec': ['single', 'double'],
        'omp': ['20'],
        'gpu': ['1'],
        'type': ['comm', 'comp', 'serial', 'other', 'total'],
        # 'N' : ['1', '16', '32'],
    }

}

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}'
plt.rcParams['font.family'] = gconfig['fontname']

if __name__ == '__main__':
    fig, ax_arr = plt.subplots(ncols=len(args.cases), nrows=1,
                               sharex=True, sharey=True,
                               figsize=gconfig['figsize'])
    ax_arr = np.atleast_1d(ax_arr)
    labels = set()
    for col, case in enumerate(args.cases):
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Reading data'))

        ax = ax_arr[col]
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
                plots_dir['_{}'.format(key)] = temp[key].copy()

        plt.grid(True, which='major', alpha=0.5, zorder=1)
        plt.grid(False, which='major', axis='x')
        plt.gca().set_axisbelow(True)

        plt.title('{}'.format(case.upper()), **gconfig['title'])
        if col == 1:
            plt.xlabel(gconfig['xlabel'], labelpad=0,
                       fontweight='bold',
                       fontsize=gconfig['fontsize'])
        if col == 0:
            plt.ylabel(gconfig['ylabel'], labelpad=0,
                       fontweight='bold',
                       fontsize=gconfig['fontsize'])

        final_dir = {}
        for key in plots_dir.keys():
            phase = key.split('_type')[1].split('_')[0]
            k = key.split('_type')[0]
            if k not in final_dir:
                final_dir[k] = {}
            if phase not in final_dir[k]:
                final_dir[k][phase] = plots_dir[key].copy()

        pos = 0
        step = 1
        width = 0.85 * step / (len(final_dir.keys()))
        print('[{}] tc: {}: {}'.format(
            this_filename[:-3], case, 'Plotting data'))
        plotted_lines = []
        for idx, k in enumerate(final_dir.keys()):
            approx = k.split('approx')[1].split('_')[0]
            red = k.split('red')[1].split('_')[0]
            experiment = k.split('_')[-1]
            prec = k.split('prec')[1].split('_')[0]
            approx = gconfig['approx'][approx]
            label = gconfig['label'][prec+approx]
            plotted_lines.append(label)
            # labels.add(label)
            bottom = []
            # colors = gconfig['colors'][idx]
            j = 0
            for phase in gconfig['phases']:

                values = final_dir[k][phase]
                y = get_values(values, header, gconfig['y_name'])
                x = get_values(values, header, gconfig['x_name'])
                omp = get_values(values, header, gconfig['omp_name'])
                if phase == 'serial':
                    y += get_values(final_dir[k]['other'],
                                    header, gconfig['y_name'])

                x = x * omp[0] // 20
                if len(x) > 1:
                    x = x[[0, -1]]
                    y = y[[0, -1]]
                else:
                    x = np.array([0, x[-1]])
                    y = np.array([0, y[-1]])
                if len(bottom) == 0:
                    bottom = np.zeros(len(y))
                print('Case: {}, version: {}, Phase: {}, Percent:'.format(case, label, phase), y)
                if phase == 'comp':
                    plt.bar(np.arange(len(x)) + pos, y, bottom=bottom, width=0.8*width,
                            label=None,
                            linewidth=1.,
                            # edgecolor=gconfig['edgecolors'][label],
                            edgecolor='black',
                            hatch=gconfig['hatches'][phase],
                            color=gconfig['colors'][label],
                            alpha=0.2,
                            zorder=2)
                else:                
                    plt.bar(np.arange(len(x)) + pos, y, bottom=bottom, width=0.8*width,
                            label=None,
                            linewidth=1.,
                            # edgecolor=gconfig['edgecolors'][label],
                            edgecolor='black',
                            hatch=gconfig['hatches'][phase],
                            color=gconfig['colors'][label],
                            alpha=gconfig['alpha'][label],
                            zorder=2)

                j += 1
                bottom += y
                if phase == 'serial':
                    for xi, yi in zip(np.arange(len(x)) + pos, bottom):
                        plt.gca().annotate('{:.1f}'.format(yi),
                                           xy=(xi, yi + 1), **gconfig['annotate'])

            pos += width
        plt.xticks(np.arange(len(x))+step/4,
                   np.array(x, int), **gconfig['ticks'])

        plt.ylim(gconfig['ylim'])
        plt.xlim(0-width, len(x)-width/2)
        if col == 0:
            ax.tick_params(**gconfig['tick_params_left'])
        else:
            ax.tick_params(**gconfig['tick_params_center_right'])

        plt.xticks(**gconfig['ticks'])
        plt.yticks(gconfig['yticks'], gconfig['yticks'], **gconfig['ticks'])

        if col == 0:
            handles = []
            for label in plotted_lines:
                patch = mpatches.Patch(label=label, edgecolor='black',
                                       facecolor=gconfig['colors'][label],
                                       linewidth=1.,)
                handles.append(patch)

            for tc, h in gconfig['hatches'].items():
                if tc =='comp':
                    alpha = 0.2
                else:
                    alpha = 1
                patch = mpatches.Patch(label=tc, edgecolor='black',
                                       facecolor='0.6', hatch=h, linewidth=1.,
                                       alpha=alpha)
                handles.append(patch)

            plt.legend(handles=handles, **gconfig['legend'])
    plt.tight_layout()
    plt.subplots_adjust(**gconfig['subplots_adjust'])
    for file in gconfig['outfiles']:
        file = file.format(images_dir, this_filename[:-3], '-'.join(args.cases))
        print('[{}] {}: {}'.format(this_filename[:-3], 'Saving figure', file))
        save_and_crop(fig, file, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()
