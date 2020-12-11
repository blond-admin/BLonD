#!/usr/bin/python
import os
import argparse
import shutil as sh

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

parser = argparse.ArgumentParser(description='Delete result directories that failed to run.',
                                 usage='python delete_broken_dirs.py.py -i [indir] -a [action]')

parser.add_argument('-i', '--indir', type=str,
                    help='The directory to look for broken runs.')

parser.add_argument('-a', '--action', type=str, default='print', choices=['rm', 'print'],
                    help='Remove or only print broken runs.')

parser.add_argument('-d', '--dontask', action='store_true',
                    help='Do not ask before deleting.')

parser.add_argument('-w', '--word', default='Done',
                   help='Word to look for in the output.txt files.'
                   '\nDefault: Done')

if __name__ == '__main__':
    args = parser.parse_args()
    to_remove = []
    for dirs, subdirs, files in os.walk(args.indir):
        if ('log' not in subdirs) or ('report' not in subdirs):
            continue
        error_str = None
        if 'output.txt' not in files:
            error_str = 'Dir: {} -- Missing output.txt'.format(dirs)
        elif ('error.txt' in files) and ('DUE TO TIME LIMIT' in open(dirs + '/error.txt').read()):
            error_str = 'Dir: {} -- Job aborted due to time limit.'.format(dirs)
        elif args.word not in open(dirs + '/output.txt').read():
            error_str = 'Dir: {} -- {} not in output.txt'.format(
                dirs, args.word)
        if error_str:
            print(error_str)
            if args.action == 'rm':
                if args.dontask:
                    ans = 'y'
                else:
                    ans = input('Delete? (Y/N) << ').lower()
                if ans in ['yes', 'y']:
                    to_remove.append(dirs)

    for directory in to_remove:
        sh.rmtree(directory)
