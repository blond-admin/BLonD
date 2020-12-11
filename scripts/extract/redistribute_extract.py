#!/usr/bin/python
import os
import csv
import sys
import fnmatch
import numpy as np
import subprocess
import argparse
import glob


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
log_fname = 'particles.csv'
log_fname_std = 'particles-std.csv'
log_worker_fname = 'particles-workers.csv'
worker_pattern = 'worker-*.log'

parser = argparse.ArgumentParser(description='Generate a csv report from the input raw data.',
                                 usage='python redistribute.py -i [indir] -o [outfile]')

parser.add_argument('-o', '--outfile', type=str, default='file',
                    choices=['sys.stdout', 'file'],
                    help='The file to save the report.'
                    ' Default: (indir)-report.csv')

parser.add_argument('-i', '--indir', type=str, default=None,
                    help='The directory containing the collected data.')

parser.add_argument('-r', '--report', type=str, default='all',
                    choices=['generate', 'collect', 'aggregate', 'all'],
                    help='The report type.')

parser.add_argument('-s', '--script', type=str, default=this_directory + 'helper_redistribute.py',
                    help='The path to the helper_redistribute script.')

parser.add_argument('-u', '--update', action='store_true',
                    help='Force update of already calculated reports.')


def generate_reports(input, report_script):
    print('\n--------Generating reports-------\n')
    records = []
    for dirs, subdirs, files in os.walk(input):
        if 'log' not in subdirs:
            continue
        ps = []
        print(dirs)
        log_dir = os.path.join(dirs, 'log')
        outfile = os.path.join(dirs, log_worker_fname)
        if (args.update or (not os.path.isfile(outfile))):
            ps.append(subprocess.Popen(['python', report_script, '-r', 'particles',
                                        '-i', log_dir, '-o', outfile,
                                        '-p', worker_pattern]))
        for p in ps:
            p.wait()


def write_avg(files, outfile, outfile_std):
    acc_data = []
    default_header = []
    data_dic = {}
    for f in files:
        dic = {}
        try:
            data = np.genfromtxt(f, dtype=str, delimiter='\t')
            header, data = data[0], data[1:]
            wids, data = data[:, 0], data[:, 1:]
            header = header[1:]  # remove the wid
        except IndexError as ie:
            print('Problem with file: ', indir + '/' + f)
            continue
            
        if len(default_header) == 0:
            default_header = header
        elif not np.array_equal(default_header, header):
            print('Problem with file: ', indir+'/'+f)
            continue
        # Get some general info
        ppb = int(f.split('_p')[1].split('_')[0])
        bunches = int(f.split('_b')[1].split('_')[0])
        workers = int(f.split('_w')[1].split('_')[0])
        parts_t0 = ppb * bunches // workers

        # go through the data column by column
        for i, h in enumerate(header):
            lst = [d[i].split('|') for d in data]
            # min_num = np.min([len(l) for l in lst])
            # lst = [l[:min_num] for l in lst]
            dic[h] = np.array(lst, float)
            if h == 'parts':
                dic[h] = np.insert(dic[h], 0, parts_t0, axis=1)
                dic[h] = np.abs(np.diff(dic[h]))/parts_t0

        for k, v in dic.items():
            if k == 'turn_num':
                if k not in data_dic:
                    data_dic[k] = []
                data_dic[k].append(np.array(v[0]))
            else:
                if k+'_avg' not in data_dic:
                    data_dic[k+'_avg'] = []
                    data_dic[k+'_min'] = []
                    data_dic[k+'_max'] = []
                data_dic[k+'_avg'].append(np.mean(v, axis=0))
                data_dic[k+'_min'].append(np.min(v, axis=0))
                data_dic[k+'_max'].append(np.max(v, axis=0))

    acc_data = []
    acc_data_std = []
    # sortid = [i[0]for i in sorted(enumerate(data_dic[funcs[-1]]),
    #                               key=lambda a:a[1][0])]
    for k, v in data_dic.items():
        # data_dic[k] = np.array(v)[sortid][:args.keep]
        # acc_data.append([k] + list(np.around(np.mean(data_dic[k], axis=0), 4)))
        acc_data.append([k] + list(np.mean(data_dic[k], axis=0)))
        acc_data_std.append(
            [k] + list(np.around(np.std(data_dic[k], axis=0), 4)))

    # Transpose list of lists magic
    acc_data = list(map(list, zip(*acc_data)))
    acc_data_std = list(map(list, zip(*acc_data_std)))
    writer1 = csv.writer(outfile, delimiter='\t')
    writer1.writerows(acc_data)
    writer2 = csv.writer(outfile_std, delimiter='\t')
    writer2.writerows(acc_data_std)


def aggregate_reports(input):
    print('\n--------Aggregating reports-------\n')

    date_pattern = '*.*-*-*'
    for dirs, subdirs, _ in os.walk(input):
        sdirs = fnmatch.filter(subdirs, date_pattern)
        if len(sdirs) == 0:
            continue
        files = [os.path.join(dirs, s, log_worker_fname) for s in sdirs]
        print(dirs)
        # try:
        write_avg(files, open(os.path.join(dirs, log_fname), 'w'),
                   open(os.path.join(dirs, log_fname_std), 'w'))
        # except Exception as e:
        # print('[Error] Dir: {}, Exception: {}, line: {}'.format(dirs, e,
        # sys.exc_info()[2].tb_lineno))


def collect_reports(input, outfile, filename):
    print('\n--------Collecting reports-------\n')
    header = ['ppb', 'bunches', 'slices', 'turns', 'n', 'omp', 'N', 'red']
    records = []
    for dirs, subdirs, files in os.walk(input):
        if filename not in files:
            continue

        print(dirs)
        try:
            config = dirs.split('/')[-1]
            ts = config.split('_t')[1].split('_')[0]
            ps = config.split('_p')[1].split('_')[0]
            bs = config.split('_b')[1].split('_')[0]
            ss = config.split('_s')[1].split('_')[0]
            ws = config.split('_w')[1].split('_')[0]
            oss = config.split('_o')[1].split('_')[0]
            Ns = config.split('_N')[1].split('_')[0]
            rs = config.split('_r')[1].split('_')[0]

            data = np.genfromtxt(os.path.join(dirs, filename),
                                 dtype=str, delimiter='\t')

            data_head, data = data[0], data[1:]
            for r in data:
                records.append([ps, bs, ss, ts, ws, oss, Ns, rs] + list(r))
        except:
            print('[Error] dir ', dirs)
            continue
    records.sort(key=lambda a: (float(a[0]), int(a[1]), int(a[2]),
                                int(a[3]), int(a[4]), int(a[5]), int(a[6])))
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(header + list(data_head))
    writer.writerows(records)


if __name__ == '__main__':
    args = parser.parse_args()

    indirs = glob.glob(args.indir)
    for indir in indirs:
        print('\n------Extracting {} -------'.format(indir))
        if args.report in ['generate', 'all']:
            generate_reports(indir, args.script)
        if args.report in ['aggregate', 'all']:
            aggregate_reports(indir)
        if args.report in ['collect', 'all']:
            if args.outfile == 'sys.stdout':
                collect_reports(indir, sys.stdout, log_fname)
            elif args.outfile == 'file':
                collect_reports(indir,
                                open(os.path.join(
                                    indir, 'particles-std-report.csv'), 'w'),
                                log_fname_std)
                collect_reports(indir,
                                open(os.path.join(
                                    indir, 'particles-report.csv'), 'w'),
                                log_fname)
