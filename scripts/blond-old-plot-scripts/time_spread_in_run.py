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
import csv

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]


parser = argparse.ArgumentParser(
    description='Normalized time spread between the various workers.',
    usage='{} infile -o outfile'.format(this_filename))

parser.add_argument('-i', '--indir', action='store', type=str,
                    help='The input directory that contains the raw data.')

parser.add_argument('-o', '--outdir', action='store', type=str,
                    default='./', help='The directory to store the plots.')

parser.add_argument('-yg', '--yamlglobal', type=str,
                    default=this_directory+'global_config.yml',
                    help='The global yaml config file.')

parser.add_argument('-yl', '--yamllocal', type=str,
                    default=this_directory+this_filename.replace('.py', '.yml'),
                    help='The local yaml config file.')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots or save only.')

# parser.add_argument('-r', '--record', action='store_true',
#                     help='Store the output data in a file.')

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # read yaml config file
    yc = yaml.load(open(args.yamlglobal, 'r'))
    globyc = yc['global']
    locyc = yaml.load(open(args.yamllocal, 'r'))
    if 'lhc' in args.indir.lower():
        testcase = 'LHC'
    elif 'sps' in args.indir.lower():
        testcase = 'SPS'
    elif 'ps' in args.indir.lower():
        testcase = 'PS'
    elif 'ex01' in args.indir.lower():
        testcase = 'EX01'
    outfiles = [
        '{}/{}-{}.jpeg'.format(args.outdir, this_filename[:-3], testcase),
        '{}/{}-{}.pdf'.format(args.outdir, this_filename[:-3], testcase)]
    recordfile = '{}/{}-{}.csv'.format(args.outdir,
                                       this_filename[:-3], testcase)
    datadic = {}
    for configdir in os.listdir(args.indir):
        if not os.path.isdir(os.path.join(args.indir, configdir)):
            continue
        parts = configdir.split('_p')[1].split('_')[0]
        bunches = configdir.split('_b')[1].split('_')[0]
        slices = configdir.split('_s')[1].split('_')[0]
        turns = configdir.split('_t')[1].split('_')[0]
        workers = int(configdir.split('_w')[1].split('_')[0])
        threads = configdir.split('_o')[1].split('_')[0]
        nodes = configdir.split('_N')[1].split('_')[0]
        red = configdir.split('_r')[1].split('_')[0]
        seed = configdir.split('_seed')[1].split('_')[0]
        approx = configdir.split('_approx')[1].split('_')[0]
        # mpiv = configdir.split('_mpi')[1].split('_')[0]
        if workers not in datadic:
            datadic[workers] = {}

        run = 0
        for rundir in os.listdir(os.path.join(args.indir, configdir)):
            if not os.path.isdir(os.path.join(args.indir, configdir, rundir)):
                continue
            reportdir = os.path.join(args.indir, configdir, rundir, 'report')
            y = {}
            for file in os.listdir(reportdir):
                if 'worker' not in file:
                    continue
                data = np.genfromtxt(os.path.join(reportdir, file),
                                     dtype=str, delimiter='\t')
                h, data = list(data[0]), data[1:]
                for phase in locyc['phases']:
                    if phase not in y:
                        y[phase] = []
                    if phase == 'total':
                        acc = np.sum([float(r[h.index('total_time(sec)')])
                                      for r in data if ':' in r[h.index('function')]])
                    else:
                        acc = np.sum([float(r[h.index('total_time(sec)')])
                                      for r in data if phase in r[h.index('function')]])
                        
                    y[phase].append(acc)
            for phase in y.keys():
                if phase not in datadic[workers]:
                    datadic[workers][phase] = {}
                # if run not in datadic[workers][phase]:
                #     datadic[workers][phase][run] = {
                #         'min': 0, 'max': 0, 'avg': 0, 'vals': []}
                # datadic[workers][phase][run]['parts'].append(int(parts)/1e6)
                # datadic[workers][phase][run]['slices'].append(int(slices))
                # datadic[workers][phase][run]['min'] = np.min(y[phase])
                # datadic[workers][phase][run]['max'] = np.max(y[phase])
                # datadic[workers][phase][run]['avg'] = np.mean(y[phase])
                # datadic[workers][phase][run]['vals'] = y[phase]
                datadic[workers][phase][run] = y[phase]
                # datadic[workers][phase][run]['ystd'].append(np.std(y[phase]))

            run += 1
    figconf = locyc['figure']
    fig, ax = plt.subplots(**figconf['figure'])
    # records = [['num_workers', 'phase', 'runid', 'deg', 'nmse']]
    labels = set()
    workers = [2, 4, 8, 12, 16]
    for i, num_workers in enumerate(workers):
        pos = 0
        w = 1 / (len(datadic[num_workers]) + 1)
        # totavg = np.mean([v for v in datadic[num_workers]['total'].values()])
        for j, phase in enumerate(datadic[num_workers].keys()):

            vals = np.array([v for v in datadic[num_workers][phase].values()])
            avgs = np.mean(vals, axis=1)
            mins = np.min(vals, axis=1)
            maxs = np.max(vals, axis=1)
            spreads = (maxs - mins) / avgs
            avg = np.mean(spreads)
            # vals = vals.flatten() / totavg
            # avg = np.mean(vals)
            # yerrlow = np.min(vals)
            # yerrhigh = np.max(vals)
            
            label=None
            if i == 0 and j == 0:
                label = 'measurement'

            # plt.errorbar([i+pos]*len(avgs), avgs/avg,
            plt.errorbar([i+pos]*len(spreads), spreads,
                         markersize=6, color='black',
                         marker='x', label=label)
            label=None
            if i == 0:
                label = phase
            # plt.bar(i+pos, 1, width=w, yerr=[[yerrlow], [yerrhigh]],
            plt.bar(i+pos, avg, width=w, 
                # yerr=[[avg-yerrlow], [yerrhigh-avg]],
                    color=figconf['colors'][phase],
                    edgecolor='0', label=label,
                    capsize=5, alpha=0.7)
            pos += w

    ax.tick_params(**figconf['tick_params'])
    plt.ylim(top=2)
    # plt.ylim(bottom=0)
    plt.title('Time spread across workers in the same run, {}'.format(testcase))
    plt.xlabel('Cores (x10)', **figconf['title'])
    plt.ylabel('Normalized Spread', **figconf['title'])
    plt.legend(**figconf['legend'])
    plt.xticks(np.arange(len(workers))+w, workers, **figconf['title'])
    plt.yticks(fontsize=8)
    plt.tight_layout()

    # fig.text(0, 0.5, 'Norm MSE', va='center',
    #          rotation='vertical', **figconf['title'])
    # fig.text(0.5, 0, 'Run ID', ha='center', **figconf['title'])
    # plt.subplots_adjust(**figconf['subplots_adjust'])
    for outfile in outfiles:
        save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()

    # if args.record:
    #     with open(recordfile, 'w') as fp:
    #         csvwriter = csv.writer(fp, delimiter='\t')
    #         # csvwriter.writerow(['kernel_name', 'all_stalls'])
    #         csvwriter.writerows(records)
