#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import matplotlib.patches as mpatches
import sys
from plot.plotting_utilities import *
from cycler import cycle
import yaml
import argparse

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

# project_dir = this_directory + '../../'
# res_dir = project_dir + 'results/'
# images_dir = res_dir + 'plots/redistribute/'

parser = argparse.ArgumentParser(
    description='Correlation plots.',
    usage='{} infile -o outfile'.format(this_filename))

parser.add_argument('-i', '--indir', action='store', type=str,
                    help='The input directory that contains the raw data.')

parser.add_argument('-o', '--outdir', action='store', type=str,
                    default='./', help='The directory to store the plots.')

parser.add_argument('-yg', '--yamlglobal', type=str,
                    default=this_directory+'configs/global_config.yml',
                    help='The global yaml config file.')

parser.add_argument('-yl', '--yamllocal', type=str,
                    default=this_directory+'configs/' + this_filename.replace('.py', '.yml'),
                    help='The local yaml config file.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots or save only.')

# Force sans-serif math mode (for axes labels)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}'
plt.rcParams['font.family'] = 'DejaVu Sans Mono'

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # read yaml config file
    yc = yaml.load(open(args.yamlglobal, 'r'), Loader=yaml.FullLoader)
    globyc = yc['global']
    locyc = yaml.load(open(args.yamllocal, 'r'), Loader=yaml.FullLoader)
    if 'lhc' in args.indir.lower():
        testcase = 'LHC'
    elif 'sps' in args.indir.lower():
        testcase = 'SPS'
    elif 'ps' in args.indir.lower():
        testcase = 'PS'
    elif 'ex01' in args.indir.lower():
        testcase = 'EX01'
    outfiles = [
        '{}/{}-{}.png'.format(args.outdir, this_filename[:-3], testcase),
        '{}/{}-{}.pdf'.format(args.outdir, this_filename[:-3], testcase)]

    datadic = {}
    for dirs, subdirs, files in os.walk(args.indir):
        if len(subdirs) > 0 or 'report' not in dirs:
            continue
        parts = dirs.split('_p')[1].split('_')[0]
        bunches = dirs.split('_b')[1].split('_')[0]
        slices = dirs.split('_s')[1].split('_')[0]
        turns = dirs.split('_t')[1].split('_')[0]
        workers = int(dirs.split('_w')[1].split('_')[0])
        threads = dirs.split('_o')[1].split('_')[0]
        nodes = dirs.split('_N')[1].split('_')[0]
        red = dirs.split('_r')[1].split('_')[0]
        seed = dirs.split('_seed')[1].split('_')[0]
        approx = dirs.split('_approx')[1].split('_')[0]
        mpiv = dirs.split('_mpi')[1].split('_')[0]
        if workers not in datadic:
            datadic[workers] = {}

        for i, file in enumerate(files):
            if i >= int(workers):
                break
            data = np.genfromtxt(os.path.join(dirs, file),
                                 dtype=str, delimiter='\t')
            h, data = list(data[0]), data[1:]
            dict = {}
            for phase in locyc['phases']:
                if phase not in datadic[workers]:
                    datadic[workers][phase] = {}
                if i not in datadic[workers][phase]:
                    datadic[workers][phase][i] = {'x': [], 'y': []}
                acc = np.sum([float(r[h.index('total_time(sec)')])
                              for r in data if phase in r[h.index('function')]])
                datadic[workers][phase][i]['x'].append(int(parts)/1e6)
                datadic[workers][phase][i]['y'].append(acc)
    figconf = locyc['figure']
    fig, ax_arr = plt.subplots(**figconf['figure'])
    # for i, num_workers in enumerate(sorted(list(datadic.keys()))):
    for i, num_workers in enumerate([4]):
        for j, phase in enumerate(['comp', 'comm', 'serial']):
            ax = ax_arr[j]
            plt.sca(ax)
            if i == 0:
                if 'serial' in phase:
                    plt.title('Intra-process', **figconf['title'], y=0.95)
                elif 'comp' in phase:
                    plt.title('Computation', **figconf['title'], y=0.95)
                else:
                    plt.title('Communication', **figconf['title'], y=0.95)
            if j == 0:
                plt.ylabel('Time (ms)', fontsize=10)
            if j == 1:
                plt.xlabel(r'Macro-particles ($10^6$)', fontsize=10)
                # plt.ylabel('W={}'.format(
                    # num_workers), **figconf['title'])
            total_x = []
            total_y = []
            total_dir = {}
            for w in datadic[num_workers][phase].keys():
                x = datadic[num_workers][phase][w]['x']
                y = datadic[num_workers][phase][w]['y']
                for xi, yi in zip(x,y):
                    if xi not in total_dir:
                        total_dir[xi] = []
                    total_dir[xi].append(yi)
                # TODO: remove some outliers
                # total_x += x
                # total_y += y
            for xi in sorted(total_dir.keys()):
                no_outliers = reject_outliers(np.array(total_dir[xi]), m=3)
                print(f'{phase}:{xi}: removed: {len(total_dir[xi]) - len(no_outliers)}/{len(total_dir[xi])}')
                total_dir[xi] = no_outliers
                total_x += [xi] * len(no_outliers)
                total_y += list(no_outliers)
            plt.scatter(total_x, total_y, marker='x', s=10, c='tab:orange')
            x = np.array(sorted(total_x))
            for deg in [1]:
                p = np.polyfit(total_x, total_y, deg=deg)
                y = 0
                label = ''
                for a, exp in zip(p, range(len(p), 0, -1)):
                    y += a * x**(exp-1)
                    label = label + '{:.1f}x^{:.0f}+'.format(a, exp-1)
                label = label[:-4]
                mse = np.mean((y - total_y)**2/y)
                plt.plot(x, y, ls='-', color='tab:blue', lw=1.5,
                         # label='{}\nNMSE={:.1f}'.format(label, mse))
                         label='NMSE: {:.2f}'.format(mse))
            ax.tick_params(**figconf['tick_params'])
            plt.legend(**figconf['legend'])
            plt.xticks(np.unique(x), fontsize=10)
            plt.yticks([0, 50, 100, 150, 200, 250], fontsize=10)
            plt.tight_layout()
            plt.grid(True, which='major', alpha=0.5)
            plt.grid(False, which='major', axis='x')
            plt.gca().set_axisbelow(True)
    # fig.text(0, 0.5, 'Time(ms)', va='center',
    #          rotation='vertical', **figconf['title'])
    # fig.text(0.5, 0, 'Particles (10^6)', ha='center', **figconf['title'])
    plt.subplots_adjust(**figconf['subplots_adjust'])
    for outfile in outfiles:
        save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()

    # for case_tup in cases:
    #     case = case_tup[0] + '-' + case_tup[1]
    #     plots_dir = {}
    #     for file, fconfig in conf['files'].items():
    #         file = file.format(res_dir, case_tup[0], case_tup[1])
    #         print(file)
    #         data = np.genfromtxt(file, delimiter='\t', dtype=str)
    #         header, data = list(data[0]), data[1:]
    #         temp = dictify(
    #             header, data, fconfig['dic_rows'], fconfig['dic_cols'])
    #         plots_dir.update(temp)

    #     fig, ax_arr = plt.subplots(**conf['subplots_args'], sharex=True)
    #     fig.suptitle(case.upper())
    #     i = 0
    #     for pltconf in conf['subplots']:
    #         ax = ax_arr[i//conf['subplots_args']['ncols'],
    #                     i % conf['subplots_args']['ncols']]
    #         plt.sca(ax)
    #         plt.title(pltconf['title'], fontsize=8)
    #         step = 1
    #         pos = 0
    #         width = step / (len(plots_dir.keys()) + 1)
    #         for nw, data in plots_dir.items():
    #             x = np.array(data[pltconf['x_name']], float)
    #             y = np.sum([np.array(data[y_name], float)
    #                         for y_name in pltconf['y_name']], axis=0)
    #             ymin = np.sum([np.array(data[y_name], float)
    #                            for y_name in pltconf['y_min_name']], axis=0)
    #             ymax = np.sum([np.array(data[y_name], float)
    #                            for y_name in pltconf['y_max_name']], axis=0)
    #             if x[0] == 0:
    #                 x, y, ymin, ymax = x[1:], y[1:], ymin[1:], ymax[1:]
    #             idx = np.linspace(0, len(x)-1, conf['points'], dtype=int)
    #             if pltconf['title'] in ['tconst', 'tcomm',
    #                                     'tcomp', 'tsync', 'ttotal', 'tpp']:
    #                 turndiff = np.diff(np.insert(x, 0, 0))
    #                 y /= turndiff
    #                 ymin /= turndiff * y[0]
    #                 ymax /= turndiff * y[0]
    #                 y /= y[0]
    #                 # ymin = np.cumsum(ymin) / x / (y[0]/x[0])
    #                 # ymax = np.cumsum(ymax) / x / (y[0]/x[0])
    #                 # y = np.cumsum(y) / x / (y[0]/x[0])
    #                 plt.axhline(1, color='k', ls='dotted', lw=1, alpha=0.5)
    #                 if np.max(ymax) > 2.:
    #                     plt.ylim(top=2.)
    #             x, y, ymin, ymax = x[idx], y[idx], ymin[idx], ymax[idx]

    #             plt.bar(np.arange(len(x)) + pos, y, width=width,
    #                     label='{}'.format(nw), lw=1, edgecolor='0',
    #                     color=conf['colors'][nw],
    #                     yerr=[y-ymin, ymax-y], error_kw={'capsize': 2, 'elinewidth': 1})
    #             pos += 1.05*width
    #             # plt.errorbar(x + displ, y, yerr=[ymin, ymax], label='{}'.format(nw),
    #             #              lw=1, capsize=1, color=conf['colors'][nw])
    #         plt.xticks(np.arange(len(x))+pos/2, np.array(x, int))
    #         ax.tick_params(**conf['tick_params'])
    #         plt.legend(**conf['legend'])
    #         plt.xticks(fontsize=8)
    #         plt.yticks(fontsize=8)
    #         plt.tight_layout()
    #         i += 1

    #     plt.subplots_adjust(**conf['subplots_adjust'])
    #     for outfile in conf['outfiles']:
    #         outfile = outfile.format(images_dir, case)
    #         save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
    #     plt.show()
    #     plt.close()
