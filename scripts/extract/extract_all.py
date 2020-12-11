import subprocess
import os
import sys
import argparse
import glob

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

parser = argparse.ArgumentParser(description='Script to extract all the collected results.',
                                 usage='python {} -i results/local/ -t lhc sps ps'.format(this_filename[:-3]))


parser.add_argument('-t', '--testcases', type=str, default='lhc,sps,ps',
                    help='A comma separated list of the testcases to run. Default: lhc,sps,ps')

parser.add_argument('-i', '--indir', type=str, default='./results/local/',
                    help='The directory with the raw data. Default: ./results/local/')

extract_script = os.path.join(this_directory, 'extract.py')

if __name__ == '__main__':
    args = parser.parse_args()
    failed_dirs = []
    for case in args.testcases.split(','):
        basedir = os.path.join(args.indir, case)
        if not os.path.isdir(basedir):
            print('[{}] No such directory: {}'.format(this_filename[:-3], basedir))
            continue
        for directory in os.listdir(basedir):
            directory = os.path.join(basedir, directory)
            if os.path.isdir(directory) and len(glob.glob(directory + '/.analysis')) > 0:
                cmd = ['python', extract_script, '--indir', directory]
                output = subprocess.run(cmd, stdout=sys.stdout,
                                        stderr=subprocess.STDOUT, env=os.environ.copy())
                if output.returncode != 0:
                    failed_dirs.append(directory)
    if failed_dirs:
        print('[{}] there was something wrong with the following directories:'.format(this_filename[:-3]))
        for f in failed_dirs:
            print(f)
