#!/usr/bin/python
import os
import csv
import sys
import fnmatch
import numpy as np
import subprocess
import argparse
import glob

'''
Instructions on how this script works

1) in each report directory (experiment/config/run/report/) there is a report 
    file each worker (worker-XX.csv).
2) The average of these files is store in exp/config/run/avg-workers.csv and
    comm-comp-workers.csv
3) The std of these files is stored in exp/config/run/avg-workers-std.csv
4) All these are done in the generate_reports phase (calling report_workers.py)
5) Then commes the aggregate phase, that reads the avg-workers.csv, avg-workers-std.csv,
and comm-comp-workers.csv from all runs and generates the: 
    * exp/config/avg.csv: The average of the averages
    * exp/config/avg-std.csv: The std of the averages
    * exp/config/avg-std-std.csv: The std of the stds
    * exp/config/avg-std-avg.csv: The average of the stds
    * exp/config/comm-comp.csv: The average of the averages
    * exp/config/comm-comp-std.csv: The std of the averages
6) Then there is the collect phase which collects the data for each config
and stores it into the files:
    * exp/avg-report.csv: All the avg.csv contents 
    * exp/avg-std-report.csv: All the avg-std.csv contents
    * exp/avg-std-avg-report.csv: All the avg-std-avg.csv contents
    * exp/avg-std-std-report.csv: All the avg-std-std.csv contents
    * exp/comm-comp-report.csv: All the comm-comp.csv contents
    * exp/comm-comp-std-report.csv: All the comm-comp-std.csv contents
'''

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

# 1st phase (generate), input
worker_pattern = 'worker-*.csv'

# 1st phase output, 2nd phase input (aggregate)
average_worker_fname = 'avg-workers.csv'
delta_worker_fname = 'delta-workers.csv'
average_std_worker_fname = 'avg-workers-std.csv'
comm_comp_worker_fname = 'comm-comp-workers.csv'

# 2nd phase output, 3rd phase input (collect)
average_fname = 'avg.csv'
average_std_fname = 'avg-std.csv'
delta_average_fname = 'delta-avg.csv'
delta_std_fname = 'delta-std.csv'
average_std_std_fname = 'avg-std-std.csv'
average_std_avg_fname = 'avg-std-avg.csv'
comm_comp_fname = 'comm-comp.csv'
comm_comp_std_fname = 'comm-comp-std.csv'

# 3rd phase output (collect)
avg_report = 'avg-report.csv'
avg_std_report = 'avg-std-report.csv'
delta_report = 'delta-report.csv'
delta_std_report = 'delta-std-report.csv'
avg_std_avg_report = 'avg-std-avg-report.csv'
avg_std_std_report = 'avg-std-std-report.csv'
comm_comp_report = 'comm-comp-report.csv'
comm_comp_std_report = 'comm-comp-std-report.csv'


parser = argparse.ArgumentParser(description='Generate a csv report from the input raw data.',
                                 usage='python extract.py -i [indir]')

parser.add_argument('-o', '--outfile', type=str, default='file',
                    choices=['sys.stdout', 'file'],
                    help='The file to save the report.'
                    ' Default: (indir)-report.csv')

parser.add_argument('-i', '--indir', type=str, default=None,
                    help='The directory containing the collected data.')

parser.add_argument('-r', '--report', type=str, default='all',
                    choices=['generate', 'collect', 'aggregate', 'all'],
                    help='The report type.')

parser.add_argument('-s', '--script', type=str, default=this_directory + 'report_workers.py',
                    help='The path to the report_workers script.')

parser.add_argument('-k', '--keep', type=int, default=None,
                    help='The number of top best runs to keep for the average calculation. Use -1 for all.')

parser.add_argument('-u', '--update', action='store_true',
                    help='Force update of already calculated reports.')

parser.add_argument('-d', '--delta', action='store_true',
                    help='Calculate the worker times deltas too.')

parser.add_argument('-check', '--check-std', type=int, default=1,
                    help='Check the STD of the extracted reports and flag configs with too high variation.'
                    'default: 1 (run the check)')

parser.add_argument('-check-std-func', '--check-std-func', nargs='+', default=['total_time'],
                    help='Functions to check for their std. A list of strings.')

parser.add_argument('-check-std-cutoff', '--check-std-cutoff', type=float, default=0.1,
                    help='Lowest value to trigger the STD report.')

parser.add_argument('-check-std-file', '--check-std-file', type=str, default=None,
                    help='File to save the reports, by default print to the std.')


args = parser.parse_args()


def generate_reports(input, report_script):
    print('\n--------Generating reports-------\n')
    for dirs, subdirs, files in os.walk(input):
        if 'report' not in subdirs:
            continue
        ps = []
        # print(dirs)
        report_dir = os.path.join(dirs, 'report')
        outfile1 = os.path.join(dirs, comm_comp_worker_fname)
        outfile2 = os.path.join(dirs, average_worker_fname)
        outfile3 = os.path.join(dirs, delta_worker_fname)
        if (args.update or (not os.path.isfile(outfile1))):
            ps.append(subprocess.Popen(['python', report_script, '-r', 'comm-comp',
                                        '-i', report_dir, '-o', outfile1,
                                        '-p', worker_pattern]))

        if (args.update or (not os.path.isfile(outfile2))):
            ps.append(subprocess.Popen(['python', report_script, '-r', 'avg',
                                        '-i', report_dir, '-o', outfile2,
                                        '-p', worker_pattern]))

        if args.delta and (args.update or (not os.path.isfile(outfile3))):
            ps.append(subprocess.Popen(['python', report_script, '-r', 'delta',
                                        '-i', report_dir, '-o', outfile3,
                                        '-p', worker_pattern]))

        for p in ps:
            p.wait()


def write_avg(files, outfile, outfile_std, check_std=False):
    acc_data = []
    default_header = []
    data_dic = {}
    for f in files:
        data = np.genfromtxt(f, dtype=str, delimiter='\t', )
        if len(data) <= 1 or len(data.shape) == 1:
            print('Empty file: ', indir+'/'+f)
            continue

        header, data = data[0], data[1:]
        funcs, data = data[:, 0], np.array(data[:, 1:], float)

        if len(default_header) == 0:
            default_header = header
        elif not np.array_equal(default_header, header):
            print('Problem with file: ', indir+'/'+f)
            continue

        for i, f in enumerate(funcs):
            if f not in data_dic:
                data_dic[f] = []
            data_dic[f].append(data[i])

    if len(data_dic) == 0:
        return

    acc_data = [default_header]
    acc_data_std = [default_header]
    sortid = [i[0]for i in sorted(enumerate(data_dic[funcs[-1]]),
                                  key=lambda a:a[1][0])]
    for f, v in data_dic.items():
        data_dic[f] = np.array(v)[sortid][:args.keep]
        acc_data.append([f] + list(np.around(np.mean(data_dic[f], axis=0), 2)))
        # The std is normalized to the mean.
        acc_data_std.append(
            [f] + list(np.around(np.abs(np.std(data_dic[f], axis=0) /
                                        np.mean(data_dic[f], axis=0)), 2))
        )

    if check_std:
        if len(files) == 1:
            print('\n[{}] WARNING: Only one input file!.\n'.format(
                os.path.dirname(files[0])), file=args.check_std_file)
        else:
            for line in acc_data_std:
                if line[0] in args.check_std_func and np.float(line[1]) > args.check_std_cutoff:
                    print('\n[{}] WARNING: STD is {}.\n'.format(
                        os.path.dirname(os.path.commonprefix(
                            (files[0], files[1]))),
                        np.float(line[1])), file=args.check_std_file)

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
        # print(dirs)

        files = [os.path.join(dirs, s, comm_comp_worker_fname) for s in sdirs]
        write_avg(files, open(os.path.join(dirs, comm_comp_fname), 'w'),
                  open(os.path.join(dirs, comm_comp_std_fname), 'w'))

        files = [os.path.join(dirs, s, average_worker_fname) for s in sdirs]
        write_avg(files, open(os.path.join(dirs, average_fname), 'w'),
                  open(os.path.join(dirs, average_std_fname), 'w'))

        files = [os.path.join(dirs, s, average_std_worker_fname) for s in sdirs]
        write_avg(files, open(os.path.join(dirs, average_std_avg_fname), 'w'),
                  open(os.path.join(dirs, average_std_std_fname), 'w'))

        if args.delta:
            files = [os.path.join(dirs, s, delta_worker_fname) for s in sdirs]
            write_avg(files, open(os.path.join(dirs, delta_average_fname), 'w'),
                      open(os.path.join(dirs, delta_std_fname), 'w'))

def collect_reports(input, outfile, filename):
    print('\n--------Collecting reports-------\n')
    # header = ['ppb', 'bunches', 'slices', 'turns', 'n', 'omp', 'N', 'red']
    records = []
    for dirs, subdirs, files in os.walk(input):
        if filename not in files:
            continue

        # print(dirs)
        try:
            config = dirs.split('/')[-1]
            ts = config.split('_t')[1].split('_')[0]
            ps = config.split('_p')[1].split('_')[0]
            bs = config.split('_b')[1].split('_')[0]
            ss = config.split('_s')[1].split('_')[0]
            ws = config.split('_w')[1].split('_')[0]
            oss = config.split('_o')[1].split('_')[0]
            Ns = config.split('_N')[1].split('_')[0]
            rs = config.split('_red')[1].split('_')[0]
            mtw = config.split('_mtw')[1].split('_')[0]
            seed = config.split('_seed')[1].split('_')[0]
            approx = config.split('_approx')[1].split('_')[0]
            prec = config.split('_prec')[1].split('_')[0]
            mpiv = config.split('_mpi')[1].split('_')[0]
            lb = config.split('_lb')[1].split('_')[0]
            artdel = config.split('_artdel')[1].split('_')[0]
            gpu = config.split('_gpu')[1].split('_')[0]
            # lba = config.split('_lba')[1].split('_')[0]
            tp = config.split('_tp')[1].split('_')[0]

            data = np.genfromtxt(os.path.join(dirs, filename),
                                 dtype=str, delimiter='\t')
            if len(data) == 0:
                print('Problem collecting data directory: ', dirs)
                continue
            data_head, data = data[0], data[1:]
            for r in data:
                records.append([ps, bs, ss, ts, ws, Ns, oss, rs,
                                mtw, seed, approx, prec, mpiv, lb, tp,
                                artdel, gpu] + list(r))
        except:
            print('[Error] dir ', dirs)
            continue
    try:
        records.sort(key=lambda a: (float(a[0]), int(a[1]), int(a[2]), int(a[4]),
                                    int(a[9]), a[11]))
        writer = csv.writer(outfile, delimiter='\t')
        header = ['ppb', 'b', 's', 't', 'n', 'N', 'omp',
                  'red', 'mtw', 'seed', 'approx', 'prec', 'mpi', 'lb', 'tp', 
                  'artdel', 'gpu'] + list(data_head)
        writer.writerow(header)
        writer.writerows(records)
        if records:
            return 0
    except Exception as e:
        print(e)
        return -1


if __name__ == '__main__':

    indirs = glob.glob(args.indir)

    if args.check_std != 0:
        if args.check_std_file is None:
            args.check_std_file = sys.stdout
        else:
            args.check_std_file = open(args.check_std_file, 'w')

    for indir in indirs:
        print('\n------Extracting {} -------'.format(indir))
        if args.report in ['generate', 'all']:
            generate_reports(indir, args.script)
        if args.report in ['aggregate', 'all']:
            aggregate_reports(indir)
        if args.report in ['collect', 'all']:
            if args.outfile == 'sys.stdout':
                collect_reports(indir, sys.stdout, average_fname)
                collect_reports(indir, sys.stdout, comm_comp_fname)
            elif args.outfile == 'file':
                errorcode = 0
                errorcode = errorcode or collect_reports(indir,
                                                         open(os.path.join(
                                                             indir, avg_std_report), 'w'),
                                                         average_std_fname)
                errorcode = errorcode or collect_reports(indir,
                                                         open(os.path.join(
                                                             indir, avg_report), 'w'),
                                                         average_fname)
                errorcode = errorcode or collect_reports(indir,
                                                         open(os.path.join(
                                                             indir, avg_std_avg_report), 'w'),
                                                         average_std_avg_fname)
                errorcode = errorcode or collect_reports(indir,
                                                         open(os.path.join(
                                                             indir, comm_comp_report), 'w'),
                                                         comm_comp_fname)
                errorcode = errorcode or collect_reports(indir,
                                                         open(os.path.join(
                                                             indir, comm_comp_std_report), 'w'),
                                                         comm_comp_std_fname)
                # For plot_all.py
                if errorcode == 0:
                    open(os.path.join(indir, '.extracted'), 'a').close()

                if args.delta:
                    collect_reports(indir,
                                    open(os.path.join(
                                        indir, delta_std_report), 'w'),
                                    delta_std_fname)
                    collect_reports(indir,
                                    open(os.path.join(indir, delta_report), 'w'),
                                    delta_average_fname)
    args.check_std_file.close()
